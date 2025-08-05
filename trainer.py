import sys
import logging
import copy
import torch
import torch.nn.functional as F
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os

# === Model Fusion Manager ===
class ModelFusion:
    def __init__(self):
        self.task_nets = []     # ذخیره مدل‌های تسک‌های قبلی
        self.task_weights = []  # وزن هر مدل بر اساس دقت

    def add_model(self, net, acc_top1):
        # ذخیره‌ی کپی از شبکه در حالت eval
        net_copy = copy.deepcopy(net).eval()
        self.task_nets.append(net_copy)
        # وزن‌دهی اولیه: استفاده از دقت top1
        self.task_weights.append(acc_top1)
        # نرمال‌سازی وزن‌ها
        total = sum(self.task_weights)
        self.task_weights = [w / total for w in self.task_weights]

    def predict(self, image, text):
        # ترکیب weighted average خروجی logits
        combined = None
        for net, w in zip(self.task_nets, self.task_weights):
            logits = net(image, text)
            if combined is None:
                combined = w * logits
            else:
                combined = combined + w * logits
        return combined

# ==============================

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "/content/drive/MyDrive/saved_models/proof/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"], args["dataset"], init_cls,
        args['increment'], args['prefix'], args['seed'], args['convnet_type'])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"],
        args["init_cls"], args["increment"]
    )
    model = factory.get_model(args["model_name"], args)
    model.save_dir = logs_name

    # ایجاد شیء Fusion پیش از حلقه تسک‌ها
    fusion = ModelFusion()

    # منحنی‌ها
    cnn_curve = {"top1": [], "top5": []}

    for task in range(min(data_manager.nb_tasks, 3)):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))

        # آموزش مدل روی تسک جدید
        model.incremental_train(data_manager)

        # ارزیابی مدل تک-تسک
        cnn_accy, nme_accy, *_ = model.eval_task()

        # افزودن مدل و وزن‌دهی بر اساس دقت top1
        fusion.add_model(model._network, cnn_accy['top1'])

        model.after_task()

        # ذخیره مدل تسک
        save_path = os.path.join(logs_name, f"model_task_{task}.pth")
        torch.save(model._network.state_dict(), save_path)
        logging.info(f"Model saved to {save_path}")

        # ثبت منحنی دقت
        cnn_curve['top1'].append(cnn_accy['top1'])
        cnn_curve['top5'].append(cnn_accy['top5'])
        logging.info("CNN top1 curve: {}".format(cnn_curve['top1']))

        print('Average Accuracy (CNN):', sum(cnn_curve['top1'])/len(cnn_curve['top1']))
        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve['top1'])/len(cnn_curve['top1'])))

    # ====== فاز ارزیابی Unified ======
    logging.info("=== Evaluating Unified Model via Fusion ===")
    total, correct = 0, 0
    # استفاده از loader تست (می‌توانید loader مناسب را از data_manager دریافت کنید)
    test_loader = data_manager.get_test_loader(args["batch_size"])
    for images, texts, labels in test_loader:
        images, texts, labels = images.to(args['device'][0]), texts.to(args['device'][0]), labels.to(args['device'][0])
        with torch.no_grad():
            fused_logits = fusion.predict(images, texts)
            preds = fused_logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    unified_acc = correct / total * 100
    logging.info(f"Unified Model Accuracy: {unified_acc:.2f}%")


def _set_device(args):
    gpus = []
    for d in args['device']:
        #device = torch.device("cuda:{}".format(d) if d >= 0 else "cpu")
        if d == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(d))
        gpus.append(device)
    args['device'] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for k, v in args.items():
        logging.info(f"{k}: {v}")
