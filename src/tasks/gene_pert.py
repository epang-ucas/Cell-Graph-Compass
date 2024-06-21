import logging

from ..data.pert_dataset import PertDataset
from unicore.tasks import UnicoreTask, register_task

logger = logging.getLogger(__name__)


# TODO add embed options
@register_task("gene_pert")
class GenePert(UnicoreTask):
    """Task for gene pertubation."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data_path", default=None)
        parser.add_argument("vocab_path", default=None)
        parser.add_argument("--use_graph", action="store_true", default=False)
        parser.add_argument("--shuffle", action="store_true", default=False)
        parser.add_argument("--embed", action="store_true", default=False)
        parser.add_argument("--text_emb_path", default=None)
        parser.add_argument("--edge_path", default=None)

    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed
        self.data_path = args.data_path
        self.vocab_path = args.vocab_path
        self.use_graph = args.use_graph
        self.shuffle = args.shuffle
        self.use_embed = args.embed
        self.text_emb_path = args.text_emb_path
        self.edge_path = args.edge_path

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        dataset = PertDataset(
            data_path = self.data_path,
            vocab_path = self.vocab_path,
            use_gnn = self.use_graph,
            use_embed = self.use_embed,
            text_emb_path = self.text_emb_path,
            split = split,
            mode = split,
            shuffle = self.shuffle,
            edge_path = self.edge_path,
        )

        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        # self.config = model.config
        return model

    def disable_shuffling(self) -> bool:
        return not self.shuffle
    
    # def begin_valid_epoch(self, epoch, model):
    #     """Hook function called before the start of each validation epoch."""
    #     if epoch % self.validate_interval_updates == 0:
    #         model.validate = True
    #     else:
    #         model.validate = False