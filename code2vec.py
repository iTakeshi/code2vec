import os
import shutil

from vocabularies import VocabType
from config import Config
from interactive_predict import InteractivePredictor
from tensorflow_model import Code2VecModel


if __name__ == '__main__':
    config = Config(set_defaults=True, load_from_args=True, verify=True)

    model = None

    if config.is_training:
        if config.OPTIMIZE:
            config.log("Start hyperparameter optimization")
            model_save_path_dir = "/".join(config.MODEL_SAVE_PATH.split("/")[:-1])
            model_save_path_name = config.MODEL_SAVE_PATH.split("/")[-1]
            for dim in range(50, 201, 10):
                save_dir = f"{model_save_path_dir}/dim_{dim}"
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.mkdir(save_dir)
                config.DEFAULT_EMBEDDINGS_SIZE = dim
                config.TOKEN_EMBEDDINGS_SIZE = config.DEFAULT_EMBEDDINGS_SIZE
                config.PATH_EMBEDDINGS_SIZE = config.DEFAULT_EMBEDDINGS_SIZE
                config.CODE_VECTOR_SIZE = config.context_vector_size
                config.TARGET_EMBEDDINGS_SIZE = config.CODE_VECTOR_SIZE
                config.MODEL_SAVE_PATH = f"{save_dir}/{model_save_path_name}"
                model = Code2VecModel(config)
                model.train()
                model.close_session()
                model = None

        else:
            model = Code2VecModel(config)
            model.train()

    if model is None:
        model = Code2VecModel(config)

    if config.is_training:
        model.train()
    if config.SAVE_W2V is not None:
        model.save_word2vec_format(config.SAVE_W2V, VocabType.Token)
        config.log('Origin word vectors saved in word2vec text format in: %s' % config.SAVE_W2V)
    if config.SAVE_T2V is not None:
        model.save_word2vec_format(config.SAVE_T2V, VocabType.Target)
        config.log('Target word vectors saved in word2vec text format in: %s' % config.SAVE_T2V)
    if (config.is_testing and not config.is_training) or config.RELEASE:
        eval_results = model.evaluate()
        if eval_results is not None:
            config.log(
                str(eval_results).replace('topk', 'top{}'.format(config.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION)))
    if config.PREDICT:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    model.close_session()
