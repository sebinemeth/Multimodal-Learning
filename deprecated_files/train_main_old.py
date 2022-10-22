import torch.utils.data as data
import torch.utils.data
from tqdm import tqdm

from utils_transforms.spatial_transforms import Compose, ToTensor, Normalize, Scale

from utils_datasets.nv_gesture.nv_dataset import NV
from utils.tensorboard_utils import *
from models.i3dpt import *
from utils_training.validation import *

video_path = "../datasets/nvGesture"
train_annotation_path = "../datasets/nvGesture/nvgesture_train_correct_cvpr2016_v2.lst"
valid_annotation_path = "../datasets/nvGesture/nvgesture_test_correct_cvpr2016_v2.lst"
model_save_dir = "./saved_models/model"
# train_path = "./datasets/senz3d_dataset"
# test_path = "./datasets/senz3d_dataset"
# train_path = "/home/sagar/data/senz3d_dataset/dataset/train/"
# test_path = "/home/sagar/data/senz3d_dataset/dataset/test/"

# TODO
# img_x, img_y = 320, 240  # resize video 2d frame size
img_x, img_y = 224, 224  # resize video 2d frame size
depth_x, depth_y = 224, 224
sample_duration = 64
# depth_x, depth_y = 320, 240
# Select which frame to begin & end in videos
# begin_frame, end_frame, skip_frame = 1, 8, 1
n_epoch = 10
only_with_gesture = True
num_classes = 25 if only_with_gesture else 25 + 1
lr = 1e-4
_lambda = 0.05  # 50 x 10^-3


def main():
    # Detect devices
    # print(torch.__version__)
    args = parse()
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # Initialize Tensorboard

    tb_writer = initialize_tensorboard(
        log_dir="./tensorboard_logs/",
        common_name="experiment{}".format(args.save_as))

    global model_save_dir
    model_save_dir += args.save_as
    os.makedirs(model_save_dir, exist_ok=True)

    train_videos_path = []
    test_videos_path = []

    # test_idx_list = [9, 10, 11]

    # for s_idx in range(1, 5):
    #     for g_idx in range(1, 12):
    #         path = os.path.join(data_path, f"S{s_idx}", f"G{g_idx}")
    #         if g_idx in test_idx_list:
    #             test_videos_path.append(path)
    #         else:
    #             train_videos_path.append(path)

    # for folder in os.listdir(train_path):
    #     train_videos_path.append(train_path + folder)
    #
    # for folder in os.listdir(test_path):
    #     test_videos_path.append(test_path + folder)

    # TODO Scale(img_x)
    # selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
    # print("len(selected_frames): {}".format(len(selected_frames)))

    norm_method = Normalize([0, 0, 0], [1, 1, 1])

    training_data = NV(video_path,
                       train_annotation_path,
                       "train",
                       num_classes,
                       frame_jump=1,
                       spatial_transform=Compose([Scale((img_x, img_y)), ToTensor(), norm_method]),
                       # temporal_transform=TemporalRandomCrop(sample_duration, downsample=1),
                       # target_transform=target_transform,
                       sample_duration=sample_duration,
                       modality="RGB-D",
                       only_with_gesture=only_with_gesture,
                       img_size=(img_x, img_y)
                       )

    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=2,
                                               shuffle=True,
                                               num_workers=10,
                                               pin_memory=True)

    valid_data = NV(video_path,
                    valid_annotation_path,
                    "valid",
                    num_classes,
                    frame_jump=1,
                    spatial_transform=Compose([Scale((img_x, img_y)), ToTensor(), norm_method]),
                    # temporal_transform=TemporalRandomCrop(sample_duration, downsample=1),
                    # target_transform=target_transform,
                    sample_duration=sample_duration,
                    modality="RGB-D",
                    only_with_gesture=only_with_gesture,
                    img_size=(img_x, img_y)
                    )

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=2,
                                               shuffle=False,
                                               num_workers=10,
                                               pin_memory=True)

    # train_rgb_set = Senz3dDataset(train_videos_path, selected_frames, to_augment=False, mode='train')
    # test_rgb_set = Senz3dDataset(test_videos_path, selected_frames, to_augment=False, mode='test')

    # train_loader = data.DataLoader(train_rgb_set, pin_memory=True, batch_size=1)
    # valid_loader = data.DataLoader(test_rgb_set, pin_memory=True, batch_size=1)

    # model_rgb_cnn = CNN3D(t_dim=sample_duration, ch_in=3, img_x=img_x, img_y=img_y, num_classes=num_classes).to(device)
    # summary(model_rgb_cnn, input_size=(sample_duration * 3, img_y, img_x))
    model_rgb_cnn = I3D(num_classes=num_classes,
                        modality='rgb',
                        dropout_prob=0,
                        name='inception').to(device)

    # model_depth_cnn = CNN3D(t_dim=sample_duration, ch_in=1, img_x=depth_x, img_y=depth_y, num_classes=num_classes).to(device)
    # summary(model_depth_cnn, input_size=(sample_duration, img_y, img_x))
    model_depth_cnn = I3D(num_classes=num_classes,
                          modality='depth',
                          dropout_prob=0,
                          name='inception').to(device)

    optimizer_rgb = torch.optim.Adam(model_rgb_cnn.parameters(), lr=lr)  # optimize all cnn parameters
    optimizer_depth = torch.optim.Adam(model_depth_cnn.parameters(), lr=lr)  # optimize all cnn parameters
    criterion = torch.nn.CrossEntropyLoss()
    args = parse()
    args.device = device

    def regularizer(loss1, loss2, beta=2):
        if loss1 - loss2 > 0:
            return (beta * math.exp(loss1 - loss2)) - 1
        return 0.0

    for epoch in range(n_epoch):
        tq = tqdm(total=(len(train_loader)))
        tq.set_description('ep {}, {}'.format(epoch, lr))
        """
        {"loss_rgb": mean_rgb, "loss_reg_rgb": mean_reg_rgb, "loss_depth": mean_depth,
            "loss_reg_depth": mean_reg_depth}
        """
        train(args=args,
              model_rgb=model_rgb_cnn,
              model_depth=model_depth_cnn,
              optimizer_rgb=optimizer_rgb,
              optimizer_depth=optimizer_depth,
              train_loader=train_loader,
              valid_loader=valid_loader,
              criterion=criterion,
              regularizer=regularizer,
              epoch=epoch,
              tb_writer=tb_writer,
              tq=tq)


if __name__ == "__main__":
    main()
