import os
import time

from six.moves import xrange
from pprint import pprint
import h5py
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pandas as pd
import random
from input_ops import create_input_ops
from util import log
from config import argparser
from GA import Genetic_algorithm

# to choose gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Trainer(object):
    def __init__(self, config, model, dataset, dataset_test):
        self.config = config
        self.model = model
        hyper_parameter_str = '{}_lr_g_{}_d_{}_update_G{}D{}'.format(
            config.dataset, config.learning_rate_g, config.learning_rate_d,
            config.update_rate, 1
        )
        self.train_dir = './train_dir/%s-%s-%s' % (
            config.prefix,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )

        os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size
        _, self.batch_train = create_input_ops(
            dataset, self.batch_size, is_training=True)
        _, self.batch_test = create_input_ops(
            dataset_test, self.batch_size, is_training=False)

        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)

        # --- checkpoint and monitoring ---
        all_var = tf.trainable_variables()

        d_var = [v for v in all_var if v.name.startswith('Discriminator')]
        slim.model_analyzer.analyze_vars(d_var, print_info=False)

        g_var = [v for v in all_var if v.name.startswith(('Generator'))]
        slim.model_analyzer.analyze_vars(g_var, print_info=False)

        rem_var = (set(all_var) - set(d_var) - set(g_var))

        self.d_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.d_loss,
            global_step=self.global_step,
            learning_rate=self.config.learning_rate_d,
            optimizer=tf.train.AdamOptimizer(beta1=0.5),
            clip_gradients=20.0,
            name='d_optimize_loss',
            variables=d_var
        )

        self.g_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.g_loss,
            global_step=self.global_step,
            learning_rate=self.config.learning_rate_g,
            optimizer=tf.train.AdamOptimizer(beta1=0.5),
            clip_gradients=20.0,
            name='g_optimize_loss',
            variables=g_var
        )

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=1000)

        self.summary_writer = tf.summary.FileWriter(self.train_dir)

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=600,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

    def GA_run_single_step(self, batch, step=None, is_train=True):

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.model.accuracy, self.summary_op,
                 self.model.d_loss, self.model.g_loss, self.model.S_loss,
                 self.model.all_preds, self.model.all_targets,
                 self.model.fake_image]

        if step % (self.config.update_rate + 1) > 0:
            # Train the generator
            fetch.append(self.g_optimizer)
        else:
            # Train the discriminator
            fetch.append(self.d_optimizer)

        fetch_values = self.session.run(fetch,
                                        feed_dict=self.model.get_feed_dict(batch_chunk, step=step)
                                        )

        [step, loss, summary, d_loss, g_loss, s_loss, all_preds, all_targets, g_img] = fetch_values[:9]

        return loss

    def run_GA(self):
        print("GA-Training Starts!")
        step = self.session.run(self.global_step)
        mean_accuracy = 0.0
        for s in range(1000):
            accuracy = self.GA_run_single_step(self.batch_train, step=s)

            mean_accuracy += accuracy
        print("run-GA Done")
        print("mean-accuracy=",mean_accuracy/1000.0)
        return mean_accuracy / 1000.0

def GA_train(population):

    for i in range(population.shape[0]):
        print("population ", i)
        print(population.ix[i])
        tf.reset_default_graph()
        config, model, dataset_train, dataset_test = argparser(is_train=True)
        config.learning_rate_g = population.ix[i]["learning_rate_g"]
        config.learning_rate_d = population.ix[i]["learning_rate_d"]
        config.update_rate = population.ix[i]["update_rate"]
        trainer = Trainer(config, model, dataset_train, dataset_test)

        population.loc[i, "accuracy_score"] = trainer.run_GA()

    return population

def GA_selection(population, GA):
    population.sort_values(ascending=False, by=["accuracy_score"], inplace=True)
    survivors = population.iloc[0:int(population.shape[0] * GA.fraction_best_kept), :].reset_index(drop=True)

    print(survivors)

    mothers = survivors.iloc[survivors.index % 2 == 0].reset_index(drop=True)
    fathers = survivors.iloc[survivors.index % 2 == 1].reset_index(drop=True)
    next_generation = pd.DataFrame(columns=["learning_rate_g", "learning_rate_d", "update_rate", "accuracy_score"])

    for i in range(population.shape[0]):
        for j in range(population.shape[1] - 1):
            rand_mom = random.randint(0, mothers.shape[0] - 1)
            rand_dad = random.randint(0, fathers.shape[0] - 1)
            Name = next_generation.columns[j]
            next_generation.loc[i, Name] = random.choice([mothers.iloc[rand_mom, j], fathers.iloc[rand_dad, j]])

            mutate = random.random()
            if mutate <= GA.mutate:
                if random.random() <= 0.5:
                    next_generation.loc[i, Name] = next_generation.loc[i, Name] + 0.1 * GA.all_possible_genes[Name][0]
                else:
                    next_generation.loc[i, Name] = next_generation.loc[i, Name] - 0.1 * GA.all_possible_genes[Name][0]
    return survivors.ix[0], next_generation

def evolution():
    GA = Genetic_algorithm()
    population = GA.initial_population()
    survivors = pd.DataFrame()
    for i in range(GA.generations):
        print("\ngeneration ", i)
        trained_population = GA_train(population)
        survivors, population = GA_selection(trained_population, GA)

    return survivors

if __name__ == '__main__':
    parameter = evolution()
    with open('parameter.txt','w') as f_write:
        f_write.write(str(parameter))
        f_write.close()

