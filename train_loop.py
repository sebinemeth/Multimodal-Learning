import torch
from tqdm import tqdm


class TrainLoop(object):
    def __init__(self, config_dict: dict):
        self.config_dict = config_dict

    def run_loop(self, args,
                 model_rgb,
                 model_depth,
                 optimizer_rgb,
                 optimizer_depth,
                 train_loader,
                 valid_loader,
                 criterion,
                 regularizer,
                 epoch,
                 tb_writer,
                 tq):
        device = args.device
        model_rgb.train()
        model_depth.train()
        rgb_losses = []
        depth_losses = []
        rgb_regularized_losses = []
        depth_regularized_losses = []
        train_result = {}

        rgb_correct = 0
        depth_correct = 0
        total = 0

        tb_batch_freq = 20

        for epoch in range(n_epoch):
            tq = tqdm(total=(len(train_loader)))
            tq.set_description('ep {}, {}'.format(epoch, lr))
            for batch_idx, (rgb, depth, y) in enumerate(train_loader):
                # distribute data to device
                rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)  # F.one_hot(y).to(device)

                optimizer_rgb.zero_grad()
                optimizer_depth.zero_grad()

                rgb_out, rgb_feature_map = model_rgb(rgb)
                depth_out, depth_feature_map = model_depth(depth)

                rgb_feature_map = rgb_feature_map.view(rgb_feature_map.shape[0], rgb_feature_map.shape[1], -1)
                rgb_feature_map_T = torch.transpose(rgb_feature_map, 1, 2)

                depth_feature_map = depth_feature_map.view(depth_feature_map.shape[0], depth_feature_map.shape[1], -1)
                depth_feature_map_T = torch.transpose(depth_feature_map, 1, 2)
                # print("RGB fmap shape :: {}".format(rgb_feature_map.shape))
                # print("RGB fmap shape T :: {}".format(rgb_feature_map_T.shape))
                # print("depth fmap shape :: {}".format(depth_feature_map.shape))
                # print("depth fmap shape T:: {}".format(depth_feature_map_T.shape))
                # torch.save(rgb_feature_map_T, "rgbFeatureMapT.pt")

                rgb_sq_ft_map = rgb_feature_map_T.squeeze()
                rgb_avg_sq_ft_map = torch.mean(rgb_sq_ft_map, 0)
                depth_sq_ft_map = depth_feature_map_T.squeeze()
                depth_avg_sq_ft_map = torch.mean(depth_sq_ft_map, 0)

                rgb_corr = torch.bmm(rgb_feature_map_T, rgb_feature_map)
                depth_corr = torch.bmm(depth_feature_map_T, depth_feature_map)
                # print("RGB correlation ::  {}".format(rgb_corr.shape))
                # print("depth correlation :: {}".format(depth_corr.shape))

                # print("RGB  ::  {}".format(rgb_out.shape))
                # print("y :: {}".format(y))

                # loss_rgb = criterion(rgb_out, torch.max(y, 1)[1])  # index of the max log-probability
                # loss_depth = criterion(depth_out, torch.max(y, 1)[1])
                loss_rgb = criterion(rgb_out, y)  # index of the max log-probability
                loss_depth = criterion(depth_out, y)
                # print("RGB loss :: {}".format(loss_rgb))
                # print("depth loss :: {}".format(loss_depth))

                focal_reg_param = regularizer(loss_rgb, loss_depth)

                """
                norm || x ||
                    Take the difference element wise
                    Square all the values
                    Add them all together
                    Take the square root
                    Multiply it with rho
                """
                corr_diff_rgb = torch.sqrt(torch.sum(torch.sub(rgb_corr, depth_corr) ** 2))
                corr_diff_depth = torch.sqrt(torch.sum(torch.sub(depth_corr, rgb_corr) ** 2))

                # loss (m,n)
                ssa_loss_rgb = focal_reg_param * corr_diff_rgb
                ssa_loss_depth = focal_reg_param * corr_diff_depth

                # total loss
                reg_loss_rgb = loss_rgb + (_lambda * ssa_loss_rgb)
                reg_loss_depth = loss_depth + (_lambda * ssa_loss_depth)

                reg_loss_rgb.backward(retain_graph=True)
                reg_loss_depth.backward()

                optimizer_rgb.step()
                optimizer_depth.step()

                rgb_losses.append(loss_rgb.item())
                depth_losses.append(loss_depth.item())
                rgb_regularized_losses.append(reg_loss_rgb.item())
                depth_regularized_losses.append(reg_loss_depth.item())

                total += y.size(0)

                _, rgb_predicted = rgb_out.max(1)
                rgb_correct += rgb_predicted.eq(y).sum().item()

                _, depth_predicted = depth_out.max(1)
                depth_correct += depth_predicted.eq(y).sum().item()

                acc_rgb = rgb_correct / total
                acc_depth = depth_correct / total

                tq.update(1)
                if batch_idx == 0:
                    train_result.update({"rgb_ft_map": rgb_avg_sq_ft_map, "depth_ft_map": depth_avg_sq_ft_map})
                tq.set_postfix(RGB_loss='{:.2f}'.format(rgb_losses[-1]),
                               regularized_rgb_loss='{:.2f}'.format(rgb_regularized_losses[-1]),
                               acc_rgb='{:.1f}%'.format(acc_rgb * 100),
                               acc_depth='{:.1f}%'.format(acc_depth * 100))

                if batch_idx % tb_batch_freq == 0:
                    mean_rgb = np.mean(rgb_losses)
                    mean_reg_rgb = np.mean(rgb_regularized_losses)
                    mean_depth = np.mean(depth_losses)
                    mean_reg_depth = np.mean(depth_regularized_losses)
                    train_result.update({"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "acc_rgb": acc_rgb,
                                         "loss_depth": mean_depth, "loss_reg_depth": mean_reg_depth,
                                         "acc_depth": acc_depth})
                    update_tensorboard_train(tb_writer=tb_writer, epoch=tb_step, train_dict=train_result)
                    update_tensorboard_image(tb_writer, tb_step, train_result)

                    tb_step += 1

                    rgb_losses = []
                    depth_losses = []
                    rgb_regularized_losses = []
                    depth_regularized_losses = []

            valid_result = validation(model_rgb=model_rgb, model_depth=model_depth, criterion=criterion,
                                      valid_loader=valid_loader, num_classes=num_classes)
            update_tensorboard_val(tb_writer=tb_writer, epoch=epoch, valid_dict=valid_result)

            torch.save(model_rgb.state_dict(), os.path.join(model_save_dir, "model_rgb_{}.pt".format(epoch)))
            torch.save(model_depth.state_dict(), os.path.join(model_save_dir, "model_depth_{}.pt".format(epoch)))
            # mean_rgb = np.mean(rgb_losses)
            # mean_reg_rgb = np.mean(rgb_regularized_losses)
            # mean_depth = np.mean(depth_losses)
            # mean_reg_depth = np.mean(depth_regularized_losses)
            # train_result.update({"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "loss_depth": mean_depth,
            #                      "loss_reg_depth": mean_reg_depth})
            # tq.set_postfix(RGB_loss='{:.5f}'.format(train_result["loss_rgb"]),
            #                regularized_rgb_loss='{:.5f}'.format(train_result["loss_reg_rgb"]))
            # update_tensorboard(tb_writer=tb_writer, epoch=epoch, train_dict=train_result, valid_dict=valid_result)
            # update_tensorboard_image(tb_writer, epoch, train_result)
            # tb_writer.flush()
