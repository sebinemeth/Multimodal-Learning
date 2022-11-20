import os
import torch
import traceback
from torchsummary import summary

from utils.log_maker import start_log_maker, write_log
from utils.arg_parser import get_config_dict, print_to_discord
from utils.discord import DiscordBot
from utils_training.get_loaders import get_loaders
from utils_training.unimodal_train_loop import UniModalTrainLoop
from utils_training.get_models import get_models
from utils.history import History
from utils.callbacks import Tensorboard, SaveModel, EarlyStopping, CallbackRunner, EarlyStopException
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType


def write_end_log(discord: DiscordBot, text: str, title: str):
    discord.send_message(fields=[{"name": title, "value": text, "inline": True}])
    write_log("training", text, title=title, print_out=True, color="red")


def main() -> History:
    start_log_maker()
    config_dict = get_config_dict()

    # Detect devices
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    write_log("init", str(device), title="used device")
    config_dict["device"] = device

    # Make save folder for the model
    model_save_dir = os.path.join(config_dict["base_dir_path"], "model")
    os.makedirs(model_save_dir, exist_ok=True)
    config_dict["model_save_dir"] = model_save_dir

    discord = DiscordBot()
    print_to_discord(discord, config_dict)

    train_loader, valid_loader = get_loaders(config_dict, ModalityType.RGB)
    exit()
    rgb_cnn, rgb_optimizer = get_models(config_dict, only_rgb=True)

    if config_dict["print_summary"]:
        summary(rgb_cnn, input_size=(3, 32, 224, 224))  # (chanel, duration, width, height)

    criterion = torch.nn.CrossEntropyLoss()
    history = History(config_dict=config_dict,
                      epoch_keys=[(SubsetType.TRAIN, ModalityType.RGB, MetricType.LOSS),
                                  (SubsetType.TRAIN, ModalityType.RGB, MetricType.ACC),
                                  (SubsetType.VAL, ModalityType.RGB, MetricType.LOSS),
                                  (SubsetType.VAL, ModalityType.RGB, MetricType.ACC)],
                      batch_keys=[(SubsetType.TRAIN, ModalityType.RGB, MetricType.LOSS),
                                  (SubsetType.TRAIN, ModalityType.RGB, MetricType.ACC)],
                      discord=discord)
    callback_runner = CallbackRunner(
        callbacks=[EarlyStopping(history=history,
                                 key=(SubsetType.VAL, ModalityType.RGB, MetricType.LOSS),
                                 patience=4, delta=0.02),
                   SaveModel(history=history,
                             model=rgb_cnn,
                             modality=ModalityType.RGB,
                             config_dict=config_dict,
                             only_best_key=(SubsetType.VAL,
                                            ModalityType.RGB,
                                            MetricType.LOSS)),
                   Tensorboard(history=history,
                               config_dict=config_dict,
                               batch_end_keys=[(SubsetType.TRAIN, ModalityType.RGB, MetricType.LOSS),
                                               (SubsetType.TRAIN, ModalityType.RGB, MetricType.ACC)],
                               epoch_end_keys=[(SubsetType.VAL, ModalityType.RGB, MetricType.LOSS),
                                               (SubsetType.VAL, ModalityType.RGB, MetricType.ACC)])])

    train_loop = UniModalTrainLoop(config_dict,
                                   rgb_cnn,
                                   rgb_optimizer,
                                   criterion,
                                   train_loader,
                                   valid_loader,
                                   history,
                                   callback_runner,
                                   discord)
    try:
        train_loop.run_loop()
    except KeyboardInterrupt:
        write_end_log(discord=discord, text="training is stopped by keyboard interrupt", title="error")
    except EarlyStopException:
        write_end_log(discord=discord, text="training is stopped by early stopping callback", title="early stopping")
    except Exception:
        write_end_log(discord=discord, text="Training is stopped with error: {}".format(traceback.format_exc()),
                      title="error")
    finally:
        callback_runner.on_training_end()

    return history


if __name__ == "__main__":
    main()
