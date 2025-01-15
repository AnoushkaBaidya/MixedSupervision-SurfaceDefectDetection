"""This code defines an end-to-end training and evaluation pipeline for a segmentation and decision model (SegDecNet) implemented using PyTorch. """


from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from models import SegDecNet
import numpy as np
import os
from torch import nn
import torch
import utils
import pandas as pd
from data.dataset_catalog import get_dataset
import random
import cv2
from config import Config
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
matplotlib.use("Agg")


LVL_ERROR = 10  # Error logging level
LVL_INFO = 5   # Info logging level
LVL_DEBUG = 1  # Debug logging level

LOG = 1  # Logs messages with a level greater than or equal to this value
SAVE_LOG = True  # Flag to save logs to a file
WRITE_TENSORBOARD = True  # Flag to enable TensorBoard logging


# EarlyStopping class monitors validation performance and stops training if improvements plateau
class EarlyStopping:
    def __init__(self, patience=10, delta=0.01):
        """
        Initialize the EarlyStopping mechanism.
        :param patience: Number of epochs to wait for improvement before stopping.
        :param delta: Minimum improvement threshold to reset the patience counter.
        """
        self.patience = patience  # Maximum epochs to wait without improvement
        self.delta = delta  # Minimum change to consider an improvement
        self.counter = 0  # Counts epochs without improvement
        self.best_loss = None  # Best observed metric value
        self.early_stop = False  # Flag indicating whether to stop training

    def __call__(self, val_metric):
        """
        Update the EarlyStopping state based on the validation metric.
        :param val_metric: Current validation metric to evaluate.
        """
        if self.best_loss is None:
            self.best_loss = val_metric  # Set the first metric as the best metric
        elif val_metric <= self.best_loss - self.delta:
            self.counter += 1  # Increment counter if no sufficient improvement
            if self.counter >= self.patience:
                self.early_stop = True  # Trigger early stopping
        else:
            self.best_loss = val_metric  # Update best metric value
            self.counter = 0  # Reset counter

# End2End class handles the complete pipeline for training and evaluation
class End2End:
    def __init__(self, cfg: Config):
        """
        Initialize the End2End pipeline with configuration settings.
        :param cfg: Configuration object containing hyperparameters and file paths.
        """
        self.cfg: Config = cfg # Configuration object
        self.storage_path: str = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET) # Path to store results

    def _log(self, message, lvl=LVL_INFO):
        """
        Log messages based on the specified logging level.
        :param message: Message to log.
        :param lvl: Logging level of the message.
        """
        n_msg = f"{self.run_name} {message}"
        if lvl >= LOG:
            print(n_msg)

    def train(self):
        """
        Execute the training process, including initialization, training, and evaluation.
        """
        self._set_results_path() # Set paths for saving results
        self._create_results_dirs() # Create directories for output storage
        self.print_run_params() # Print and log training parameters

        if self.cfg.REPRODUCIBLE_RUN:
            # Ensure reproducibility by setting random seed
            self._log("Reproducible run, fixing all seeds to:1337", LVL_DEBUG)
            np.random.seed(1337)
            torch.manual_seed(1337)
            random.seed(1337)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Initialize device, model, optimizer, and loss functions
        device = self._get_device()
        model = self._get_model().to(device)
        optimizer = self._get_optimizer(model)
        loss_seg, loss_dec = self._get_loss(True), self._get_loss(False)

        # Load training and validation datasets
        train_loader = get_dataset("TRAIN", self.cfg)
        validation_loader = get_dataset("VAL", self.cfg)

        # Load training and validation datasets
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(self.tensorboard_path, "train_logs")) if WRITE_TENSORBOARD else None

        #early_stopper = EarlyStopping(patience=10, delta=0.01)

        # Train the model and save results
        train_results = self._train_model(device, model, train_loader, loss_seg, loss_dec, optimizer, validation_loader, tensorboard_writer)
        self._save_train_results(train_results) # Save training loss and validation metrics
        self._save_model(model) # Save the trained model

        # Evaluate the model on the test dataset
        self.eval(model, device, self.cfg.SAVE_IMAGES, False, False)
        # Save configuration parameters to disk
        self._save_params()


    def eval(self, model, device, save_images, plot_seg, reload_final):
        """
        Evaluate the model on the test dataset.
        :param model: Trained model.
        :param device: Device to use for evaluation (CPU or GPU).
        :param save_images: Whether to save predicted images.
        :param plot_seg: Whether to plot segmentation results.
        :param reload_final: Whether to reload the final model state.
        """
        self.reload_model(model, reload_final) # Load model weights
        test_loader = get_dataset("TEST", self.cfg) # Load the test dataset
        self.eval_model(device, model, test_loader, save_folder=self.outputs_path, save_images=save_images, is_validation=False, plot_seg=plot_seg)


    def training_iteration(self, data, device, model, criterion_seg, criterion_dec, optimizer, weight_loss_seg, weight_loss_dec,
                           tensorboard_writer, iter_index):
        """
        Perform a single training iteration.
        :param data: Batch of training data.
        :param device: Device to use for computation (CPU or GPU).
        :param model: Model to train.
        :param criterion_seg: Loss function for segmentation.
        :param criterion_dec: Loss function for decision/classification.
        :param optimizer: Optimizer for model parameters.
        :param weight_loss_seg: Weight for segmentation loss.
        :param weight_loss_dec: Weight for decision loss.
        :param tensorboard_writer: TensorBoard writer for logging (optional).
        :param iter_index: Index of the current iteration.
        """                  
        images, seg_masks, seg_loss_masks, is_segmented, _ = data  # Unpack input data


        batch_size = self.cfg.BATCH_SIZE # Total batch size
        # Sub-batch size for memory optimization
        memory_fit = self.cfg.MEMORY_FIT  # Not supported yet for >1 
        # Number of sub-iterations per batch
        num_subiters = int(batch_size / memory_fit)

        total_loss = 0
        total_correct = 0
        total_loss_seg = 0 # Total segmentation loss
        total_loss_dec = 0 # Total decision loss

        optimizer.zero_grad()  # Clear optimizer gradients

        for sub_iter in range(num_subiters):
            # Slice sub-batch data for processing
            images_ = images[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_masks_ = seg_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_loss_masks_ = seg_loss_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            is_pos_ = seg_masks_.max().reshape((memory_fit, 1)).to(device) # Determine positive samples
            
            # Log sample images and masks to TensorBoard
            if tensorboard_writer and iter_index % 100 == 0:
                tensorboard_writer.add_image(f"{iter_index}/image", images_[0].cpu(), dataformats="CHW")
                tensorboard_writer.add_image(f"{iter_index}/seg_mask", seg_masks[0].cpu(), dataformats="CHW")
                tensorboard_writer.add_image(f"{iter_index}/seg_loss_mask", seg_loss_masks_[0].cpu(), dataformats="CHW")

            # Forward pass
            decision, output_seg_mask = model(images_)

            if is_segmented[sub_iter]:
                # Compute segmentation loss
                if self.cfg.WEIGHTED_SEG_LOSS:
                    # Resize both seg_masks_ and seg_loss_masks_ to match output_seg_mask
                    seg_masks_resized = F.interpolate(seg_masks_, size=output_seg_mask.shape[2:], mode='bilinear', align_corners=False)
                    seg_loss_masks_resized = F.interpolate(seg_loss_masks_, size=output_seg_mask.shape[2:], mode='bilinear', align_corners=False)
                    # Compute the segmentation loss
                    loss_seg = torch.mean(criterion_seg(output_seg_mask, seg_masks_resized) * seg_loss_masks_resized)
                else:
                    loss_seg = criterion_seg(output_seg_mask, seg_masks_)

                # Compute decision loss
                loss_dec = criterion_dec(decision, is_pos_)

                # Update total losses and accuracy
                total_loss_seg += loss_seg.item()
                total_loss_dec += loss_dec.item()
                total_correct += (decision > 0.0).item() == is_pos_.item()

                # Weighted combination of losses
                loss = weight_loss_seg * loss_seg + weight_loss_dec * loss_dec

            else:
                # Compute decision loss for non-segmented samples
                loss_dec = criterion_dec(decision, is_pos_)
                total_loss_dec += loss_dec.item()  
                total_correct += (decision > 0.0).item() == is_pos_.item()
                loss = weight_loss_dec * loss_dec

            total_loss += loss.item() # Accumulate total loss
            loss.backward() # Backpropagation

        optimizer.step()  # Update model weights
        optimizer.zero_grad()  # Clear gradients

        return total_loss_seg, total_loss_dec, total_loss, total_correct

    def _train_model(self, device, model, train_loader, criterion_seg, criterion_dec, optimizer, validation_set, tensorboard_writer):
        """
        Train the model over multiple epochs while tracking losses, accuracy, and validation metrics.
        :param device: Device to use (CPU or GPU).
        :param model: The model to train.
        :param train_loader: DataLoader for the training set.
        :param criterion_seg: Loss function for segmentation.
        :param criterion_dec: Loss function for classification/decision.
        :param optimizer: Optimizer for model parameters.
        :param validation_set: DataLoader for the validation set.
        :param tensorboard_writer: Writer for TensorBoard logs (optional).
        """
        losses = [] # Track training losses
        validation_data = [] # Track validation performance
        max_validation = -1 # Highest observed validation metric
        validation_step = self.cfg.VALIDATION_N_EPOCHS  # Frequency of validation

        num_epochs = self.cfg.EPOCHS # Total number of training epochs
        samples_per_epoch = len(train_loader) * self.cfg.BATCH_SIZE # Total samples processed per epoch

        self.set_dec_gradient_multiplier(model, 0.0) # Initialize decision gradient multiplier
        early_stopper = EarlyStopping(patience=10, delta=0.01) # Early stopping monitor

        for epoch in range(num_epochs):
            self._log(f"Starting Epoch {epoch + 1}/{num_epochs}", LVL_INFO)

            model.train()  # Set model to training mode
            weight_loss_seg, weight_loss_dec = self.get_loss_weights(epoch) # Dynamic loss weighting
            dec_gradient_multiplier = self.get_dec_gradient_multiplier()
            self.set_dec_gradient_multiplier(model, dec_gradient_multiplier) # Update gradient multiplier

            epoch_loss_seg, epoch_loss_dec, epoch_loss = 0, 0, 0  # Initialize loss accumulators
            epoch_correct = 0 # Initialize accuracy counter

            from timeit import default_timer as timer
            time_acc = 0 # Time for forward and backward passes
            start = timer() # Start epoch timer

            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as t:
                for iter_index, data in t:
                    # Measure time for each iteration
                    start_1 = timer()

                    # Perform a single training iteration
                    curr_loss_seg, curr_loss_dec, curr_loss, correct = self.training_iteration(
                        data, device, model, criterion_seg, criterion_dec, optimizer,
                        weight_loss_seg, weight_loss_dec, tensorboard_writer, (epoch * samples_per_epoch + iter_index))

                    # Log iteration loss
                    print(f"Epoch {epoch + 1}, Iter {iter_index + 1}: Loss = {curr_loss:.5f}")
                    
                    end_1 = timer()
                    time_acc = time_acc + (end_1 - start_1)  # Accumulate iteration time

                    epoch_loss_seg += curr_loss_seg
                    epoch_loss_dec += curr_loss_dec
                    epoch_loss += curr_loss
                    epoch_correct += correct

                    # Update progress bar with losses
                    t.set_postfix(loss=curr_loss, seg_loss=curr_loss_seg, dec_loss=curr_loss_dec)
            end = timer() # End epoch timer

            # Average losses over all samples
            epoch_loss_seg /= samples_per_epoch
            epoch_loss_dec /= samples_per_epoch
            epoch_loss /= samples_per_epoch
            losses.append((epoch_loss_seg, epoch_loss_dec, epoch_loss, epoch))
            
            # Log epoch summary
            self._log(
                f"Epoch {epoch + 1}/{num_epochs} ==> avg_loss_seg={epoch_loss_seg:.5f}, avg_loss_dec={epoch_loss_dec:.5f}, avg_loss={epoch_loss:.5f}, correct={epoch_correct}/{samples_per_epoch}, in {end - start:.2f}s/epoch (fwd/bck in {time_acc:.2f}s/epoch)")

            print( f"Epoch {epoch + 1}/{num_epochs} ==> avg_loss_seg={epoch_loss_seg:.5f}, avg_loss_dec={epoch_loss_dec:.5f}, avg_loss={epoch_loss:.5f}, correct={epoch_correct}/{samples_per_epoch}")

            # Log metrics to TensorBoard
            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/Train/segmentation", epoch_loss_seg, epoch)
                tensorboard_writer.add_scalar("Loss/Train/classification", epoch_loss_dec, epoch)
                tensorboard_writer.add_scalar("Loss/Train/total", epoch_loss, epoch)
                tensorboard_writer.add_scalar("Accuracy/Train/", epoch_correct / samples_per_epoch, epoch)

            # Perform validation at regular intervals or on the last epoch
            if self.cfg.VALIDATE and (epoch % validation_step == 0 or epoch == num_epochs - 1):
                validation_ap, validation_accuracy, validation_AUROC, validation_F1Score = self.eval_model(device, model, validation_set, None, False, True, False)
                validation_data.append((validation_ap, epoch))
                
                # Save the best model based on validation metric
                if validation_ap > max_validation:
                    max_validation = validation_ap
                    self._save_model(model, "best_state_dict.pth")
                model.train()   # Switch back to training mode
                
                # Log validation accuracy to TensorBoard
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("Accuracy/Validation/", validation_accuracy, epoch)
                    tensorboard_writer.add_scalar("AUROC/Validation/", validation_AUROC, epoch)
                    tensorboard_writer.add_scalar("F1Score/Validation/", validation_F1Score, epoch)
                    
  
                # Check for early stopping
                early_stopper(validation_ap)
                if early_stopper.early_stop:
                    self._log(f"Early stopping at epoch {epoch + 1}.", LVL_INFO)
                    break

        return losses, validation_data


    def eval_model(self, device, model, eval_loader, save_folder, save_images, is_validation, plot_seg):
        """
        Evaluate the model on a dataset and compute performance metrics.
        :param device: Device to use (CPU or GPU).
        :param model: Model to evaluate.
        :param eval_loader: DataLoader for the evaluation dataset.
        :param save_folder: Path to save predictions (optional).
        :param save_images: Whether to save predicted images.
        :param is_validation: Whether this is a validation run (affects metrics).
        :param plot_seg: Whether to generate segmentation plots.
        """
        model.eval()  # Set model to evaluation mode

        dsize = self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT  # Desired image dimensions

        res = []  # Store results
        predictions, ground_truths = [], []  # Store predictions and ground truth labels

        for data_point in eval_loader:
            # Unpack data
            image, seg_mask, seg_loss_mask, _, sample_name = data_point
            image, seg_mask = image.to(device), seg_mask.to(device)
            is_pos = (seg_mask.max() > 0).reshape((1, 1)).to(device).item() # Determine if sample is positive

            # Forward pass
            prediction, pred_seg = model(image)
            pred_seg = nn.Sigmoid()(pred_seg)
            prediction = nn.Sigmoid()(prediction)

            # Convert outputs to usable formats
            prediction = prediction.item()
            image = image.detach().cpu().numpy()
            pred_seg = pred_seg.detach().cpu().numpy()
            seg_mask = seg_mask.detach().cpu().numpy()

            # Store predictions and ground truth
            predictions.append(prediction)
            ground_truths.append(is_pos)
            res.append((prediction, None, None, is_pos, sample_name[0]))
            if not is_validation:
                if save_images:
                    image = cv2.resize(np.transpose(image[0, :, :, :], (1, 2, 0)), dsize)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    pred_seg = cv2.resize(pred_seg[0, 0, :, :], dsize) if len(pred_seg.shape) == 4 else cv2.resize(pred_seg[0, :, :], dsize)
                    seg_mask = cv2.resize(seg_mask[0, 0, :, :], dsize)
                    if self.cfg.WEIGHTED_SEG_LOSS:
                        seg_loss_mask = cv2.resize(seg_loss_mask.numpy()[0, 0, :, :], dsize)
                        utils.plot_sample(sample_name[0], image, pred_seg, seg_loss_mask, save_folder, decision=prediction, plot_seg=plot_seg)
                    else:
                        utils.plot_sample(sample_name[0], image, pred_seg, seg_mask, save_folder, decision=prediction, plot_seg=plot_seg)
        
        # Compute and log metrics for validation
        if is_validation:
            metrics = utils.get_metrics(np.array(ground_truths), np.array(predictions))
            FP, FN, TP, TN = list(map(sum, [metrics["FP"], metrics["FN"], metrics["TP"], metrics["TN"]]))
            self._log(f"VALIDATION || AUC={metrics['AUC']:f}, and AP={metrics['AP']:f}, with best thr={metrics['best_thr']:f} "
                      f"at f-measure={metrics['best_f_measure']:.3f} and FP={FP:d}, FN={FN:d}, TOTAL SAMPLES={FP + FN + TP + TN:d}")
            
            auroc = metrics["AUC"]  # Assuming AUROC is returned as AUC
            f1_score = metrics["best_f_measure"]  # Assuming best F1 score is returned

            return metrics["AP"], metrics["accuracy"], auroc, f1_score
        else:
            utils.evaluate_metrics(res, self.run_path, self.run_name)

    def get_dec_gradient_multiplier(self):
        if self.cfg.GRADIENT_ADJUSTMENT:
            grad_m = 0
        else:
            grad_m = 1

        self._log(f"Returning dec_gradient_multiplier {grad_m}", LVL_DEBUG)
        return grad_m

    def set_dec_gradient_multiplier(self, model, multiplier):
        model.set_gradient_multipliers(multiplier)

    def get_loss_weights(self, epoch):
        total_epochs = float(self.cfg.EPOCHS)

        if self.cfg.DYN_BALANCED_LOSS:
            seg_loss_weight = 1 - (epoch / total_epochs)
            dec_loss_weight = self.cfg.DELTA_CLS_LOSS * (epoch / total_epochs)
        else:
            seg_loss_weight = 1
            dec_loss_weight = self.cfg.DELTA_CLS_LOSS

        self._log(f"Returning seg_loss_weight {seg_loss_weight} and dec_loss_weight {dec_loss_weight}", LVL_DEBUG)
        return seg_loss_weight, dec_loss_weight

    def reload_model(self, model, load_final=False):
        if self.cfg.USE_BEST_MODEL:
            path = os.path.join(self.model_path, "best_state_dict.pth")
            model.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
        elif load_final:
            path = os.path.join(self.model_path, "final_state_dict.pth")
            model.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
        else:
            self._log("Keeping same model state")

    def _save_params(self):
        params = self.cfg.get_as_dict()
        params_lines = sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", params.items()))
        fname = os.path.join(self.run_path, "run_params.txt")
        with open(fname, "w+") as f:
            f.writelines(params_lines)

    def _save_train_results(self, results):
        losses, validation_data = results
        ls, ld, l, le = map(list, zip(*losses))
        plt.plot(le, l, label="Loss", color="red")
        plt.plot(le, ls, label="Loss seg")
        plt.plot(le, ld, label="Loss dec")
        plt.ylim(bottom=0)
        plt.grid()
        plt.xlabel("Epochs")
        if self.cfg.VALIDATE:
            v, ve = map(list, zip(*validation_data))
            plt.twinx()
            plt.plot(ve, v, label="Validation AP", color="Green")
            plt.ylim((0, 1))
        plt.legend()
        plt.savefig(os.path.join(self.run_path, "loss_val"), dpi=200)

        df_loss = pd.DataFrame(data={"loss_seg": ls, "loss_dec": ld, "loss": l, "epoch": le})
        df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

        if self.cfg.VALIDATE:
            df_loss = pd.DataFrame(data={"validation_data": ls, "loss_dec": ld, "loss": l, "epoch": le})
            df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

    def _save_model(self, model, name="final_state_dict.pth"):
        output_name = os.path.join(self.model_path, name)
        self._log(f"Saving current model state to {output_name}")
        if os.path.exists(output_name):
            os.remove(output_name)

        torch.save(model.state_dict(), output_name)

    def _get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), self.cfg.LEARNING_RATE)

    def _get_loss(self, is_seg):
        reduction = "none" if self.cfg.WEIGHTED_SEG_LOSS and is_seg else "mean"
        return nn.BCEWithLogitsLoss(reduction=reduction).to(self._get_device())

    def _get_device(self):
        if torch.cuda.is_available() and self.cfg.GPU != -1:
            return f"cuda:{self.cfg.GPU}"
        else:
            return "cpu"

    def _set_results_path(self):
        self.run_name = f"{self.cfg.RUN_NAME}_FOLD_{self.cfg.FOLD}" if self.cfg.DATASET in ["KSDD", "DAGM"] else self.cfg.RUN_NAME

        results_path = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)
        self.tensorboard_path = os.path.join(results_path, "tensorboard", self.run_name)

        run_path = os.path.join(results_path, self.cfg.RUN_NAME)
        if self.cfg.DATASET in ["KSDD", "DAGM"]:
            run_path = os.path.join(run_path, f"FOLD_{self.cfg.FOLD}")

        self._log(f"Executing run with path {run_path}")

        self.run_path = run_path
        self.model_path = os.path.join(run_path, "models")
        self.outputs_path = os.path.join(run_path, "test_outputs")

    def _create_results_dirs(self):
        list(map(utils.create_folder, [self.run_path, self.model_path, self.outputs_path, ]))

    def _get_model(self):
        seg_net = SegDecNet(self._get_device(), self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_CHANNELS)
        return seg_net

    def print_run_params(self):
        for l in sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", self.cfg.get_as_dict().items())):
            k, v = l.split(":")
            self._log(f"{k:25s} : {str(v.strip())}")
