import os
import traceback
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils.log_maker import start_log_maker, write_log
from utils.arg_parser import get_config_dict, print_to_discord
from utils.discord import DiscordBot
from utils_training.get_loaders import get_loaders
from utils_training.unimodal_train_loop import UniModalTrainLoop
from utils_training.get_models import get_models
from utils.history import History
from utils.callbacks import EarlyStopping, CallbackRunner, EarlyStopException


def main() -> History:
    start_log_maker()
    config_dict = get_config_dict()

    # Detect devices
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    write_log("init", str(device), title="used device")
    config_dict["device"] = device

    # Initialize Tensorboard
    tb_log_path = os.path.join(config_dict["base_dir_path"], "tensorboard_logs")
    os.makedirs(tb_log_path, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_path)

    # Make save folder for the model
    model_save_dir = os.path.join(config_dict["base_dir_path"], "model")
    os.makedirs(model_save_dir, exist_ok=True)
    config_dict["model_save_dir"] = model_save_dir

    discord = DiscordBot()
    print_to_discord(discord, config_dict)

    history = History(keys=["rgb_loss", "rgb_acc", "val_rgb_loss", "val_rgb_acc"])
    callback_runner = CallbackRunner(callbacks=[EarlyStopping(key="val_rgb_loss", patience=4, delta=0.02)], history=history)

    train_loader, valid_loader = get_loaders(config_dict)
    rgb_cnn, rgb_optimizer = get_models(config_dict, only_rgb=True)

    if config_dict["print_summary"]:
        summary(rgb_cnn, input_size=(3, 32, 224, 224))  # (chanel, duration, width, height)

    criterion = torch.nn.CrossEntropyLoss()

    train_loop = UniModalTrainLoop(config_dict,
                                   rgb_cnn,
                                   rgb_optimizer,
                                   criterion,
                                   train_loader,
                                   valid_loader,
                                   history,
                                   callback_runner,
                                   tb_writer,
                                   discord)
    try:
        train_loop.run_loop()
    except KeyboardInterrupt:
        write_log("training", "training is stopped by keyboard interrupt", title="error", print_out=True, color="red")
        discord.send_message(fields=[{"name": "Error",
                                      "value": "Training is stopped by keyboard interrupt",
                                      "inline": True}])
    except EarlyStopException:
        write_log("training", "training is stopped by early stopping callback", title="earlystopping", print_out=True,
                  color="red")
        discord.send_message(fields=[{"name": "Early stop",
                                      "value": "Training is stopped by early stopping callback",
                                      "inline": True}])
    except Exception:
        discord.send_message(fields=[{"name": "Error",
                                      "value": "Training is stopped with error: {}".format(traceback.format_exc()),
                                      "inline": True}])
        write_log("training", "training is stopped with error:\n{}".format(traceback.format_exc()), title="error",
                  print_out=True, color="red")
    finally:
        train_loop.save_models("end")

    return history


if __name__ == "__main__":
    main()

