import os
import yaml
import argparse
from tqdm import trange, tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Model.optimizer import ScheduledOptim

from utils.utils import load_loss, log_writer, to_device, to_device_eval
from utils.model import mel_plot, synth_one_sample
from utils.model import get_param_num, load_models, load_dataset

def train_and_eval(train_config, preprocessed_config, model_config, restore_steps = None):
    if torch.cuda.is_available() :
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs(train_config["path"]["ckpt_path"], exist_ok= True)
    os.makedirs(train_config["path"]["log_path"], exist_ok=True)
    os.makedirs(train_config["path"]["result_path"], exist_ok=True)

    model_name = train_config["model"]
    model = load_models(model_name, preprocessed_config, model_config)
    model = model.to(device)
    print("{} has loaded".format(model_name))
    n_paramters = get_param_num(model)
    print("{} Parameters : {}".format(model_name, n_paramters))
    loss_func = load_loss(model_name, train_config, preprocess_config, model_config)
    if restore_steps is not None:
        optimizer = ScheduledOptim(model, train_config, model_config, restore_steps)
    else:
        optimizer = ScheduledOptim(model, train_config, model_config, 0)
    #load data loader
    #train data
    train_data = load_dataset(model_name, "train.txt", train_config, model_config, preprocess_config)
    #evaluation data
    eval_data = load_dataset(model_name, "val.txt", train_config, model_config, preprocess_config)

    #writers
    train_writer = SummaryWriter(log_dir = os.path.join(train_config["path"]["log_path"], "train"))
    eval_writer = SummaryWriter(log_dir = os.path.join(train_config["path"]["log_path"], "eval"))

    #training
    s_epoch = 0
    e_epochs = train_config["step"]["total_epochs"]
    log_steps = train_config["step"]["log_step"]
    eval_steps = train_config["step"]["synth_step"]
    global step
    if restore_steps is not None:
        s_epoch = restore_steps
        step = restore_steps * len(train_data)
    else:
        step = 0

    outer_bar = tqdm(total = e_epochs, desc = "Total_Epoches", position = 0, leave=True)
    if restore_steps is not None:
        outer_bar.update(restore_steps)

    for epoch in range(s_epoch, e_epochs):
        training(model_name = model_name, model = model, dataset = train_data, optimizer = optimizer, loss_func = loss_func, writer = train_writer, log_steps = log_steps, eval_steps = eval_steps, device = device)
        evaluation(model_name=model_name, model=model, dataset =eval_data,  loss_func=loss_func,
              writer=eval_writer, device=device)
        torch.save(
            {
                "model" : model.state_dict(),
                "optimizer" : optimizer._optimizer.state_dict()
            },
            os.path.join(train_config["path"]["ckpt_path"],
                         "{}.pth.tar".format(epoch))
        )
        outer_bar.update(1)

def training(model_name, model, dataset, optimizer, loss_func, writer, log_steps, eval_steps, device = None):
    global step
    model.train()

    inner_bar = tqdm(total=len(dataset), desc="Training", position=1)
    inner_bar.update()
    for i, data in enumerate(dataset):
        data = to_device(model_name, data, device)
        output = model(*(data[1:]))
        loss = loss_func(data, output)
        loss[0].backward()
        grad_norm = nn.utils.clip_grad_norm(model.parameters(), 1.0)

        optimizer.step_and_update_lr()
        optimizer.zero_grad()

        if step % log_steps == 0 :
            log_writer(writer, "scalar", "Loss", loss, step)
            writer.add_scalar("grad_norm", grad_norm, step)
            print("Training Total steps {} : {}, Mel loss : {}, Postnet Loss : {}, Pitch Loss : {}, Energy Loss : {}, Duration Loss : {}".format(
                step, loss[0], loss[1],loss[2],loss[3],loss[4],loss[5]
            ))
        if step % eval_steps == 0 :
            with torch.no_grad():
                ori_mel_fig = mel_plot(data[5][0].transpose(-1,-2).cpu().detach(),"Original Mel Spectrogram", data[6][0])
                ori_audio = synth_one_sample(data[5][0],model_config, preprocess_config, device, data[6][0])
                data = to_device_eval(model_name, data, device)
                output = model(*(data[1:]))
                mel_feature = output[1][0]
                gen_mel_fig = mel_plot(mel_feature.transpose(-1,-2).cpu().detach(),"Synthesized Mel Spectrogram", output[9][0])
                gen_audio_fig = synth_one_sample(mel_feature, model_config, preprocess_config, device, output[9][0])
                log_writer(writer, "image", "train/mel_original", ori_mel_fig, step)
                log_writer(writer, "image", "train/mel_gen", gen_mel_fig, step)
                log_writer(writer, "audio", "train/audio_reconstruct", ori_audio, step)
                log_writer(writer, "audio", "train/audio_gen", gen_audio_fig, step)
        step += 1
        inner_bar.update(1)

def evaluation(model_name, model, dataset, loss_func, writer, device = None):
    global step
    model.eval()

    inner_bar = tqdm(total=len(dataset), desc="Evaluation", position=1)
    inner_bar.update()

    n_losses = [0 for _ in range(6)]
    with torch.no_grad():
        for i, data in enumerate(dataset):
            data = to_device(model_name, data, device)
            output = model(*(data[1:]))
            loss = loss_func(data, output)
            for j in range(6):
                n_losses[j] = n_losses[j] + loss[j]/len(dataset)
            if i == len(dataset) - 1:
                ori_mel_fig = mel_plot(data[5][0].transpose(-1, -2).cpu().detach(), "Original Mel Spectrogram", data[6][0])
                ori_audio = synth_one_sample(data[5][0], model_config, preprocess_config, device, data[6][0])
                data = to_device_eval(model_name, data, device)
                output = model(*(data[1:]))
                mel_feature = output[1][0]
                gen_mel_fig = mel_plot(mel_feature.transpose(-1, -2).cpu().detach(), "Synthesized Mel Spectrogram", output[9][0])
                gen_audio = synth_one_sample(mel_feature, model_config, preprocess_config, device, output[9][0])
                log_writer(writer, "image", "eval/mel_original", ori_mel_fig, step)
                log_writer(writer, "image", "eval/mel_gen", gen_mel_fig, step)
                log_writer(writer, "audio", "eval/audio_reconstruct", ori_audio, step)
                log_writer(writer, "audio", "eval/audio_gen", gen_audio, step)

            inner_bar.update(1)

    log_writer(writer, "scalar", "Loss", n_losses, step)
    print("Evaluation Total steps {} : {}, Mel loss : {}, Postnet Loss : {}, Pitch Loss : {}, Energy Loss : {}, Duration Loss : {}".format(
        step, n_losses[0], n_losses[1], n_losses[2], n_losses[3], n_losses[4], n_losses[5]
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=None)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)

    train_and_eval(train_config, preprocess_config , model_config, restore_steps= args.restore_step)

