import os
import warnings
import traceback
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torchsummary import summary
from typing import Dict

from utils.log_maker import start_log_maker, write_log
from utils.arg_parser import get_config_dict, print_to_discord
from utils.discord import DiscordBot
from utils_training.get_loaders import get_loaders
from utils_training.train_loop import TrainLoop
from utils_training.get_models import get_models
from utils.history import History
from utils.callbacks import Tensorboard, SaveModel, EarlyStopping, CallbackRunner, EarlyStopException
from utils_datasets.nv_gesture.nv_utils import SubsetType, ModalityType, MetricType, NetworkType

# to supress pytorch nn.MaxPool3d warning
warnings.filterwarnings("ignore", category=UserWarning)


def main() -> History:
    start_log_maker()
    config_dict = get_config_dict()
    convert_modalities_and_network(config_dict)

    # Detect devices
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    write_log("init", "device: {}".format(str(device)), title="used device", print_out=True, color="yellow")
    config_dict["device"] = device

    # Make save folder for the model
    model_save_dir = os.path.join(config_dict["base_dir_path"], "model")
    os.makedirs(model_save_dir, exist_ok=True)
    config_dict["model_save_dir"] = model_save_dir

    discord = DiscordBot()
    print_to_discord(discord, config_dict)

    train_loader, valid_loader = get_loaders(config_dict)
    model_dict, optimizer_dict = get_models(config_dict)

    if config_dict["print_summary"]:
        for modality, model in model_dict.items():
            chanel_size = 3 if modality == ModalityType.RGB else 1
            summary(model, input_size=(chanel_size, 32, 224, 224))  # (chanel, duration, width, height)

    criterion = torch.nn.CrossEntropyLoss()
    history = get_history(config_dict, discord)
    callback_runner = get_callback_runner(model_dict, optimizer_dict, history, config_dict)

    train_loop = TrainLoop(config_dict,
                           model_dict,
                           optimizer_dict,
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


def write_end_log(discord: DiscordBot, text: str, title: str):
    discord.send_message(fields=[{"name": title, "value": text, "inline": True}])
    write_log("training", text, title=title, print_out=True, color="red")


def convert_modalities_and_network(config_dict: dict):
    if config_dict["network"] == "DETECTOR":
        config_dict["network"] = NetworkType.DETECTOR
    elif config_dict["network"] == "CLASSIFIER":
        config_dict["network"] = NetworkType.CLASSIFIER
    else:
        raise ValueError("unknown modality: {}".format(config_dict["network"]))

    modalities = list()
    for modality in config_dict["modalities"]:
        if modality == "RGB":
            modalities.append(ModalityType.RGB)
        elif modality == "DEPTH":
            modalities.append(ModalityType.DEPTH)
        else:
            raise ValueError("unknown modality: {}".format(modality))

    config_dict["modalities"] = modalities


def get_history(config_dict: dict, discord: DiscordBot) -> History:
    epoch_keys = list()
    batch_keys = list()
    for modality in config_dict["modalities"]:
        epoch_keys.extend([(SubsetType.TRAIN, modality, MetricType.LOSS),
                           (SubsetType.TRAIN, modality, MetricType.ACC),
                           (SubsetType.VAL, modality, MetricType.LOSS),
                           (SubsetType.VAL, modality, MetricType.ACC)])
        batch_keys.extend([(SubsetType.TRAIN, modality, MetricType.LOSS),
                           (SubsetType.TRAIN, modality, MetricType.ACC)])

        if len(config_dict["modalities"]) > 1:
            epoch_keys.append((SubsetType.TRAIN, modality, MetricType.REG_LOSS))
            batch_keys.append((SubsetType.TRAIN, modality, MetricType.REG_LOSS))

    history = History(config_dict=config_dict,
                      epoch_keys=epoch_keys,
                      batch_keys=batch_keys,
                      discord=discord)
    return history


def get_callback_runner(model_dict: Dict[ModalityType, Module],
                        optimizer_dict: Dict[ModalityType, Optimizer],
                        history: History,
                        config_dict: dict) -> CallbackRunner:
    callback_list = [EarlyStopping(
        history=history,
        config_dict=config_dict,
        key=(SubsetType.VAL,
             config_dict["modalities"][0] if len(config_dict["modalities"]) == 1 else ModalityType.RGB_DEPTH,
             MetricType.LOSS),
        patience=config_dict["patience"],
        delta=config_dict["delta"])]

    batch_end_keys = list()
    epoch_end_keys = list()
    for modality in config_dict["modalities"]:
        callback_list.append(SaveModel(history=history,
                                       model=model_dict[modality],
                                       optimizer=optimizer_dict[modality],
                                       modality=modality,
                                       config_dict=config_dict,
                                       only_best_key=(SubsetType.VAL, modality, MetricType.LOSS)))
        batch_end_keys.extend([(SubsetType.TRAIN, modality, MetricType.LOSS),
                               (SubsetType.TRAIN, modality, MetricType.ACC)])
        epoch_end_keys.extend([(SubsetType.VAL, modality, MetricType.LOSS),
                               (SubsetType.VAL, modality, MetricType.ACC)])

        if len(config_dict["modalities"]) > 1:
            batch_end_keys.append((SubsetType.TRAIN, modality, MetricType.REG_LOSS))
            epoch_end_keys.append((SubsetType.TRAIN, modality, MetricType.REG_LOSS))

    callback_list.append(Tensorboard(history=history,
                                     config_dict=config_dict,
                                     batch_end_keys=batch_end_keys,
                                     epoch_end_keys=epoch_end_keys))
    return CallbackRunner(callbacks=callback_list)


if __name__ == "__main__":
    main()
