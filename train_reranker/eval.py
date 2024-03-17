from sentence_transformers.evaluation import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
from sentence_transformers.util import batch_to_device
import os
import csv
from sentence_transformers import CrossEncoder
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)

class MSEEval(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, 
                 dataloader: DataLoader, 
                 name: str = "", 
                 show_progress_bar: bool = True,
                 write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.show_progress_bar = show_progress_bar

        if name:
            name = "_"+name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    def __call__(self, model: CrossEncoder, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.model.eval()
        total = 0
        loss_total = 0

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        loss_fnc = torch.nn.BCEWithLogitsLoss()
        activation_fnc = torch.nn.Identity()

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for features, labels in tqdm(self.dataloader,  desc="Evaluation", smoothing=0.05, disable=not self.show_progress_bar):
            
            with torch.no_grad():
                model_predictions = model.model(**features, return_dict=True)
                logits = activation_fnc(model_predictions.logits)
                if model.config.num_labels == 1:
                    logits = logits.view(-1)
                loss_value = loss_fnc(logits, labels)

            total += 1 # number of batches
            loss_total += loss_value.cpu().item()
        mse = loss_total/total

        logger.info("MSE: {:.4f} ({}/{})\n".format(mse, loss_total, total))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, mse])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, mse])

        return mse
    
