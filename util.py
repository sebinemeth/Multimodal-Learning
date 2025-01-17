import argparse
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def initialize_tensorboard(log_dir, common_name):
    """
    In distributed training, tensorboard doesn't work with multiple writers
    reference: https://stackoverflow.com/a/37411400/4569025
    """
    tb_log_path = Path(log_dir).joinpath(common_name)
    if not os.path.exists(tb_log_path):
        os.mkdir(tb_log_path)
    tb_writer = SummaryWriter(log_dir=str(tb_log_path))
    return tb_writer


def update_tensorboard_train(tb_writer, epoch, train_dict):
    """
    {"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "loss_depth": mean_depth,
        "loss_reg_depth": mean_reg_depth, "rgb_ft_map": rgb_avg_sq_ft_map, "depth_ft_map": depth_avg_sq_ft_map}
    """

    tb_writer.add_scalar(tag='RGB train loss', scalar_value=train_dict["loss_rgb"], global_step=epoch)
    tb_writer.add_scalar(tag='RGB regularized train loss', scalar_value=train_dict["loss_reg_rgb"], global_step=epoch)
    tb_writer.add_scalar(tag='RGB train accuracy', scalar_value=train_dict["acc_rgb"],
                         global_step=epoch)
    # tb_writer.add_scalars(main_tag='Depth train',
    #                       tag_scalar_dict={'Depth train loss': train_dict["loss_depth"],
    #                                        'Depth regularized train loss': train_dict["loss_reg_depth"]},
    #                       global_step=epoch)
    tb_writer.add_scalar(tag='Depth train loss', scalar_value=train_dict["loss_depth"], global_step=epoch)
    tb_writer.add_scalar(tag='Depth regularized train loss', scalar_value=train_dict["loss_reg_depth"],
                         global_step=epoch)
    tb_writer.add_scalar(tag='Depth train accuracy', scalar_value=train_dict["acc_depth"],
                         global_step=epoch)


def update_tensorboard_val(tb_writer, epoch, valid_dict):
    """
    {'valid_rgb_loss': valid_rgb_loss, 'valid_depth_loss': valid_depth_loss}
    """
    tb_writer.add_scalar(tag='RGB valid loss', scalar_value=valid_dict["valid_rgb_loss"], global_step=epoch)
    tb_writer.add_scalar(tag='Depth valid loss', scalar_value=valid_dict["valid_depth_loss"], global_step=epoch)

    tb_writer.add_scalar(tag='RGB valid accuracy', scalar_value=valid_dict["valid_rgb_acc"], global_step=epoch)
    tb_writer.add_scalar(tag='Depth valid accuracy', scalar_value=valid_dict["valid_depth_acc"], global_step=epoch)


def update_tensorboard_image(tb_writer, epoch, train_dict):
    tb_writer.add_image(tag='RGB feature map', img_tensor=train_dict['rgb_ft_map'].unsqueeze(dim=0), global_step=epoch)
    tb_writer.add_image(tag='Depth feature map', img_tensor=train_dict['depth_ft_map'].unsqueeze(dim=0),
                        global_step=epoch)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-as", metavar='FOLDER_NAME', required=True)
    args = parser.parse_args()
    # config = configparser.ConfigParser()
    return args
