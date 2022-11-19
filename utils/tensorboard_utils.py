def update_tensorboard_train(tb_writer, global_step, train_dict, only_rgb=False):
    """
    {"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "loss_depth": mean_depth,
        "loss_reg_depth": mean_reg_depth, "rgb_ft_map": rgb_avg_sq_ft_map, "depth_ft_map": depth_avg_sq_ft_map}
    """

    tb_writer.add_scalar(tag='RGB train loss', scalar_value=train_dict["loss_rgb"], global_step=global_step)
    tb_writer.add_scalar(tag='RGB train accuracy', scalar_value=train_dict["acc_rgb"],
                         global_step=global_step)
    # tb_writer.add_scalars(main_tag='Depth train',
    #                       tag_scalar_dict={'Depth train loss': train_dict["loss_depth"],
    #                                        'Depth regularized train loss': train_dict["loss_reg_depth"]},
    #                       global_step=epoch)
    if not only_rgb:
        tb_writer.add_scalar(tag='RGB regularized train loss', scalar_value=train_dict["loss_reg_rgb"],
                             global_step=global_step)

        tb_writer.add_scalar(tag='Depth train loss', scalar_value=train_dict["loss_depth"], global_step=global_step)
        tb_writer.add_scalar(tag='Depth regularized train loss', scalar_value=train_dict["loss_reg_depth"],
                             global_step=global_step)
        tb_writer.add_scalar(tag='Depth train accuracy', scalar_value=train_dict["acc_depth"],
                             global_step=global_step)


def update_tensorboard_val(tb_writer, global_step, valid_dict, only_rgb=False):
    """
    {'valid_rgb_loss': valid_rgb_loss, 'valid_depth_loss': valid_depth_loss}
    """
    tb_writer.add_scalar(tag='RGB valid loss', scalar_value=valid_dict["valid_rgb_loss"], global_step=global_step)
    tb_writer.add_scalar(tag='RGB valid accuracy', scalar_value=valid_dict["valid_rgb_acc"], global_step=global_step)

    if not only_rgb:
        tb_writer.add_scalar(tag='Depth valid loss', scalar_value=valid_dict["valid_depth_loss"], global_step=global_step)
        tb_writer.add_scalar(tag='Depth valid accuracy', scalar_value=valid_dict["valid_depth_acc"], global_step=global_step)


def update_tensorboard_image(tb_writer, epoch, train_dict):
    tb_writer.add_image(tag='RGB feature map', img_tensor=train_dict['rgb_ft_map'].unsqueeze(dim=0), global_step=epoch)
    tb_writer.add_image(tag='Depth feature map', img_tensor=train_dict['depth_ft_map'].unsqueeze(dim=0),
                        global_step=epoch)
