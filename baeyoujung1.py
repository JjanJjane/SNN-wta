def convert_tb_data(root_dir, sort_by=None):

    import os
    import pandas as pd
    import tensorflow as tf
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):

        return dict(
            wall_time=tfevent.wall_time,
            tag=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=tf.make_ndarray(tfevent.summary.value[0].tensor)
        )

    columns_order = ['wall_time', 'tag', 'step', 'value'] # 1

    out = []

    # parsing directory

    output_dir = 'output' #2
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #output_root_dir = os.path.join(output_dir, os.path.basename(root_dir))
    #print(root_dir) #root_dir = ~/tensorboard
    #print(os.path.basename(root_dir)) # basename(root_dir) = tensorboard
    #print(output_root_dir) # output_root_dir = output/tensorboard
    #shutil.copytree(root_dir, output_root_dir)

    for root, dirs, files in os.walk(root_dir):

        for filename in files:
            file_full_path = os.path.join(root,filename)
            df = convert_tfevent(file_full_path)
            unique_tags = df['tag'].unique()

            for tag_name in unique_tags:

                if tag_name not in ['epoch_acc', 'epoch_acc-5', 'epoch_best_acc_val', 'epoch_best_s_count',
                                   'epoch_best_val_acc', 'epoch_learning_rate', 'epoch_loss', 'epoch_s_count',
                                   'evaluation_acc_vs_iterations', 'evaluation_acc-5_vs_iterations',
                                   'evaluation_loss_vs_iterations'] : # 3
                    continue

                output_folder = os.path.join(
                    'output',
                    os.path.relpath(root, root_dir),
                    os.path.splitext(filename)[0],
                    tag_name
                )

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                #os.makedirs(output_folder, exist_ok=True)
                output_filename = os.path.join(output_folder, f"{filename}_{tag_name}.csv")

                # Filter data by tag
                tag_data = df[df['tag'] == tag_name]

                #tag_data.drop(columns=['tag'], inplace=True)

                # Save to CSV inside tag directory
                tag_data.to_csv(output_filename, index=False, columns = ['step', 'value']) #4

    # Concatenate (and sort) all partial individual dataframes
    #all_df = pd.concat(out)[columns_order]

    #return all_df.reset_index()


if __name__ == "__main__":

    dir_path = "/home/baeyoujung1/PycharmProjects/TensorFlow-SNN-internal/80_Tensorboard_SNN_Training/src/Neuro/"
    #dir_path = "/home/baeyoujung1/PycharmProjects/TensorFlow-SNN-internal/80_Tensorboard_SNN_Training/Neuro/tensorboard/221231_train_snn_VGG16_CIFAR100/VGG16_CIFAR10/"

    df = convert_tb_data(f"{dir_path}")