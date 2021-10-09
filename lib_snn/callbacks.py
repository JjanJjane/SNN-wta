
import os
import shutil

import tensorflow as tf

from tensorflow.python.keras.utils.io_utils import path_to_string


class ManageSavedModels(tf.keras.callbacks.Callback):
    def __init__(self,
                 filepath,
                 max_to_keep=5,
                 **kwargs):
        super(ManageSavedModels, self).__init__()

        self.filepath = path_to_string(filepath)
        self.max_to_keep = max_to_keep

    #
    def check_and_remove(self):
        list_dir = os.listdir(self.filepath)

        if len(list_dir) <= self.max_to_keep:
            return

        mtime = lambda f: os.stat(os.path.join(self.filepath, f)).st_mtime
        list_dir_sorted = list(sorted(os.listdir(self.filepath), key=mtime))

        for d in list_dir_sorted[:-5]:
            target_d = os.path.join(self.filepath,d)
            if os.path.isfile(target_d):
                os.remove(target_d)
            else:
                shutil.rmtree(target_d)


    #def on_train_batch_end(self, batch, logs=None):
        #self.check_and_remove()

    def on_epoch_end(self, epoch, logs=None):
        self.check_and_remove()



# ModelCheckpointResume
# wrapper for keras.callback.ModelCheckpoint
# add "best" argument for resume training
class ModelCheckpointResume(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,
                filepath,
                monitor = 'val_loss',
                verbose = 0,
                save_best_only = False,
                save_weights_only = False,
                mode = 'auto',
                save_freq = 'epoch',
                options = None,
                best = None,
                tensorboard_writer = None,
                log_dir = None,
                ** kwargs):

        if save_freq is not 'epoch':
            assert False, 'only supported save_freq=epoch'

        super(ModelCheckpointResume, self).__init__(
            filepath=filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            options=options,
            **kwargs)

        if best is not None:
            self.best = best

        #tf.summary.create_file_writer()

        #print('ModelCheckpointResume - init - previous best: '.format(self.best))


    # from keras.callbacks.ModelCheckpoint
    def on_epoch_end(self, epoch, logs=None):

        super(ModelCheckpointResume, self).on_epoch_end(epoch=epoch,logs=logs)

        #print(self.best)
        #tf.summary.scalar('best_acc_val', data=self.best, step=epoch)
        logs['best_acc_val'] = self.best


#
class TensorboardBestValAcc(tf.keras.callbacks.Callback):
    def __init__(self,
                 best_val_acc,
                 **kwargs):

        self.best_val_acc = best_val_acc
        super(TensorboardBestValAcc, self).__init__(**kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        print('on_epoch_begin')
        print(logs)

    def on_epoch_end(self, epoch, logs=None):
        print('best val_acc')
        #print(cb_model_checkpoint.best)
        print(self.best_val_acc)
        print(logs)





########
# callback test
class CallbackTest(tf.keras.callbacks.Callback):
    def __init__(self,
                 **kwargs):
        super(CallbackTest, self).__init__(
            **kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        print('on_epoch_begin')
        print(logs)

    def on_epoch_end(self, epoch, logs=None):
        print('on_epoch_begin')
        print(logs)
