
import os
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import yaml
import glob
import sys
from itertools import compress

import numpy as np
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import selfies
from tqdm import tqdm
from fpdf import FPDF
from chemvae import mol_utils as mu
from chemvae import hyperparameters
from chemvae.models import load_encoder, load_decoder, load_property_predictor
from chemvae.mol_utils import fast_verify, verify_smiles
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools, Draw
from rdkit import RDLogger

from bbox_model import BBoxModel

class VAE_Model(BBoxModel):
    # === STANDARD CONSTRUCTOR ===
    def __init__(self,
                 exp_file='exp.json',
                 encoder_file=None,
                 decoder_file=None,
                 directory=None):
        # files
        if directory is not None:
            curdir = os.getcwd()
            fpath = os.path.join(curdir, directory)

        # load parameters
        os.chdir(fpath)
        self.params = hyperparameters.load_params(exp_file, False)
        if encoder_file is not None:
            self.params["encoder_weights_file"] = encoder_file
        if decoder_file is not None:
            self.params["decoder_weights_file"] = decoder_file
        # char stuff
        chars = yaml.safe_load(open(self.params['char_file']))
        self.chars = chars
        self.params['NCHARS'] = len(chars)
        #self.char_indices = dict((c, i) for i, c in enumerate(chars))
        #self.indices_char = dict((i, c) for i, c in enumerate(chars))
        # encoder, decoder
        self.enc = load_encoder(self.params)
        self.dec = load_decoder(self.params)
        self.encode, self.decode = self._enc_dec_functions()
        self.data = None
        if self.params['do_prop_pred']:
            self.property_predictor = load_property_predictor(self.params)
        sys.stdout.write("Standardization of latent space...")
        sys.stdout.flush()  # for printing messages inside the function.
        normal_file = "normalization_info.pkl" # for preventing time-consuming
        # normalization step if it's once done.

        if os.path.isfile(normal_file):
            d_nor = pk.load(open(normal_file,'rb'))
            self.mu = d_nor['mu']
            self.std = d_nor['std']
            self.totsize = d_nor['data_size']
        else:
            # Load data without normalization as dataframe
            df = pd.read_csv(self.params['data_file'])
            df.iloc[:, 0] = df.iloc[:, 0].str.strip()
            df = df[df.iloc[:, 0].str.len() <= self.params['MAX_LEN']]
            self.smiles = df.iloc[:, 0].tolist()
            self.data_size = len(self.smiles)
            if df.shape[1] > 1:
                self.data = df.iloc[:, 1:]

            self.estimate_standardization()
            d_nor = {'mu': self.mu, 'std': self.std, 'data_size': self.data_size}
            with(open(normal_file,'wb')) as n_file:
                pk.dump(d_nor, n_file)
        sys.stdout.write('done\n')
        sys.stdout.flush()

        if directory is not None:
            os.chdir(curdir)
        return

    # === AUXILIARY METHODS ===
    def _enc_dec_functions(self, standardized=True):
        print('Using standardized representations? {}'.format(standardized))
        if not self.params['do_tgru']:
            def decode(z, standardized=standardized):
                if standardized:
                    return self.dec.predict(self.unstandardize_z(z))
                else:
                    return self.dec.predict(z)
        else:
            def decode(z, standardize=standardized):
                fake_shape = (z.shape[0], self.params[
                    'MAX_LEN'], self.params['NCHARS'])
                fake_in = np.zeros(fake_shape)
                if standardize:
                    return self.dec.predict([self.unstandardize_z(z), fake_in])
                else:
                    return self.dec.predict([z, fake_in])

        def encode(X, standardized=standardized):
            if standardized:
                return self.standardize_z(self.enc.predict(X)[0])
            else:
                return self.enc.predict(X)[0]

        return encode, decode

    def _prep_mol_df(self, smiles, z, verbose):
        RDLogger.DisableLog('rdApp.*')
        df = pd.DataFrame({'smiles': smiles})
        sort_df = pd.DataFrame(df[['smiles']].groupby(
            by='smiles').size().rename('count').reset_index())
        df = df.merge(sort_df, on='smiles')
        if verbose:
            print("How many duplicated smiles? " + str(df.duplicated().sum()) +
                  " out of " + str(len(df)))
        df.drop_duplicates(subset='smiles', inplace=True)
        verflag = df['smiles'].apply(verify_smiles)
        if verbose:
            print("How many not verified? " + str(len(df) - verflag.sum()) +
                  " out of " + str(len(df)))
        df = df[verflag]
        if not len(df) > 0:
            if verbose:
                print("No molecule was found either due to invalidity or "
                      "not being verified!")
            return df
        else:
            df['mol'] = df['smiles'].apply(mu.smiles_to_mol)
            df = df[pd.notnull(df['mol'])]
            df['distance'] = self.smiles_distance_z(df['smiles'], z)
            df['frequency'] = df['count'] / float(sum(df['count']))
            df = df[['smiles', 'distance', 'count', 'frequency', 'mol']]
            df.sort_values(by='distance', inplace=True)
            df.reset_index(drop=True, inplace=True)
            df = self._get_props_for_samples(df)
        return df

    def _get_props_for_samples(self, mol_dataframe):
        df = mol_dataframe
        cols = []
        Z = np.zeros((len(df), self.params['hidden_dim']))
        if 'reg_prop_tasks' in self.params:
            cols += self.params['reg_prop_tasks']
        if 'logit_prop_tasks' in self.params:
            cols += self.params['logit_prop_tasks']
        smiles = df['smiles']
        one_hot = self._smiles_to_hot(smiles, self.params['SELFIES'])
        Z = self.encode(one_hot)
        props = self.predict_prop_z(Z)
        df[cols] = pd.DataFrame(props, index=df.index)
        # re-ordering columns of dataframe
        new_cols = ['distance'] + cols + ['count', 'frequency', 'smiles',
                                          'mol']
        df = df.reindex(columns = new_cols)
        return df

    def _predict_property_function(self):
        # Now reports predictions after un-normalization.
        def predict_prop(X):
            # both regression and logistic
            if (('reg_prop_tasks' in self.params) and
                (len(self.params['reg_prop_tasks']) > 0) and
                    ('logit_prop_tasks' in self.params) and
                    (len(self.params['logit_prop_tasks']) > 0)):
                reg_pred, logit_pred = self.property_predictor.predict(
                    self.encode(X))
                if 'data_normalization_out' in self.params:
                    df_norm = pd.read_csv(
                        self.params['data_normalization_out'])
                    reg_pred = reg_pred * \
                               df_norm['std'].values + df_norm['mean'].values
                return reg_pred, logit_pred
            # regression only scenario
            elif (('reg_prop_tasks' in self.params) and
                  (len(self.params['reg_prop_tasks']) > 0)):
                reg_pred = self.property_predictor.predict(self.encode(X))
                if 'data_normalization_out' in self.params:
                    df_norm = pd.read_csv(
                        self.params['data_normalization_out'])
                    reg_pred = reg_pred * \
                               df_norm['std'].values + df_norm['mean'].values
                return reg_pred
            # logit only scenario
            else:
                logit_pred = self.property_predictor.predict(self.encode(X))
                return logit_pred

        return predict_prop

    def _smiles_to_hot(self, smiles, selfies=True):

        if isinstance(smiles, str):
            smiles = [smiles]
        p = self.params
        if selfies:
            selfies_list,_,_,_,_,_ = \
                mu.get_selfie_and_smiles_encodings_for_dataset(smiles)
            z = mu.multiple_selfies_to_hot(selfies_list, p['MAX_LEN'],
                                           self.chars)
        else:
            z = mu.multiple_smile_to_hot(smiles, p['MAX_LEN'],
                                           self.chars)
        return z

    def _hot_to_smiles(self, hot_x, strip=False):
        smiles = mu.hot_to_smiles(hot_x, self.chars, self.params['SELFIES'])
        if strip:
            smiles = [s.strip() for s in smiles]
        return smiles

    def _random_idxs(self, size=None):
        if size is None:
            return [i for i in range(len(self.smiles))]
        else:
            return random.sample([i for i in range(len(self.smiles))], size)

    def _random_molecules(self, size=None):
        if size is None:
            return self.smiles
        else:
            return random.sample(self.smiles, size)

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # === MANIPULATION METHODS ===
    def estimate_standardization(self):
        # sample Z space

        ssize = int(self.data_size * 0.66)
        smiles = self._random_molecules(size=ssize)
        batch = int(ssize/200)
        Z = np.zeros((len(smiles), self.params['hidden_dim']))
        for chunk in self.chunks(list(range(len(smiles))), batch):
            sub_smiles = [smiles[i] for i in chunk]
            one_hot = self._smiles_to_hot(sub_smiles, self.params['SELFIES'])
            Z[chunk, :] = self.encode(one_hot, False)

        self.mu = np.mean(Z, axis=0)
        self.std = np.std(Z, axis=0)
        self.Z = self.standardize_z(Z)
        return

    # === PROPERTY EVALUATION METHODS ===
    def standardize_z(self, z):
        return (z - self.mu) / self.std

    def unstandardize_z(self, z):
        return (z * self.std) + self.mu

    def smiles_to_z(self, smiles, standardized=True):
        x = self._smiles_to_hot(smiles, self.params['SELFIES'])
        z_rep = self.encode(x, standardized)
        return z_rep

    def z_to_smiles(self, z, standardized=True, verified=True):
        s_hot = self.decode(z, standardized)
        x_hat = self._hot_to_smiles(s_hot, strip=True)
        if verified:
            verflag = map(verify_smiles, x_hat)
            x_hat = list(map(lambda a, b: a if b==True else None,
                             x_hat, verflag))
        return x_hat

    def smiles_to_prop(self, smiles, standardized=True):
        z_mat = self.smiles_to_z(smiles, standardized)
        pred_prop_all = self.predict_prop_z(z_mat)
        return pred_prop_all

    def perturb_z(self, z, noise_norm, constant_norm=False):
        if noise_norm > 0.0:
            noise_vec = np.random.normal(0, 1, size=z.shape)
            noise_vec = noise_vec / np.linalg.norm(noise_vec)
            if constant_norm:
                return z + (noise_norm * noise_vec)
            else:
                noise_amp = np.random.uniform(
                    0, noise_norm, size=(z.shape[0], 1))
                return z + (noise_amp * noise_vec)
        else:
            return z

    def smiles_distance_z(self, smiles, z0):
        x = self._smiles_to_hot(smiles, self.params['SELFIES'])
        z_rep = self.encode(x)
        return np.linalg.norm(z0 - z_rep, axis=1)

    def sample_and_dec_to_smiles(self,
                    z,
                    decode_attempts=250,
                    noise_norm=0.0,
                    constant_norm=False,
                    early_stop=None,
                    verbose=True):
        if not (early_stop is None):
            Z = np.tile(z, (25, 1))
            Z = self.perturb_z(Z, noise_norm, constant_norm)
            X = self.decode(Z)
            smiles = self._hot_to_smiles(X, strip=True)
            df = self._prep_mol_df(smiles, z)
            if len(df) > 0:
                low_dist = df.iloc[0]['distance']
                if low_dist < early_stop:
                    return df

        Z = np.tile(z, (decode_attempts, 1))
        Z = self.perturb_z(Z, noise_norm)
        X = self.decode(Z)
        smiles = self._hot_to_smiles(X, strip=False)
        df = self._prep_mol_df(smiles, z, verbose)
        return df

    def sample_ls_to_smiles(self,
                    z,
                    num_samples=250,
                    noise_norm=0.0,
                    constant_norm=False,
                    verified=True):
        Z = np.tile(z, (num_samples, 1))
        Z = self.perturb_z(Z, noise_norm)
        X = self.decode(Z)
        xhat = self._hot_to_smiles(X, strip=True)
        if verified:
            verflag = list(map(verify_smiles, xhat))
            smiles = [s for i, s in enumerate(xhat) if verflag[i]==True]
        return smiles

    def predict_prop_z(self, z, standardized=True):
        # Now reports predictions after un-normalization.
        if standardized:
            z = self.unstandardize_z(z)
        # both regression and logistic
        if (('reg_prop_tasks' in self.params) and
            (len(self.params['reg_prop_tasks']) > 0) and
                ('logit_prop_tasks' in self.params) and
                (len(self.params['logit_prop_tasks']) > 0)):

            reg_pred, logit_pred = self.property_predictor.predict(z)
            if 'data_normalization_out' in self.params:
                df_norm = pd.read_csv(self.params['data_normalization_out'])
                reg_pred = reg_pred * \
                           df_norm['std'].values + df_norm['mean'].values
            return reg_pred, logit_pred
        # regression only scenario
        elif (('reg_prop_tasks' in self.params) and
              (len(self.params['reg_prop_tasks']) > 0)):
            reg_pred = self.property_predictor.predict(z)
            if 'data_normalization_out' in self.params:
                df_norm = pd.read_csv(self.params['data_normalization_out'])
                reg_pred = reg_pred * \
                           df_norm['std'].values + df_norm['mean'].values
            return reg_pred
        # logit only scenario
        else:
            logit_pred = self.property_predictor.predict(z)
            return logit_pred

    # === BASIC QUERY METHODS ===
    @property
    def ls_dim(self):
        return self.params['hidden_dim']

    @property
    def n_props(self):
        regp = len(self.params['reg_prop_tasks']) if 'reg_prop_tasks' in \
                self.params else 0
        regl = len(self.params['logit_prop_tasks']) if 'logit_prop_tasks' in \
                self.params else 0
        return regp + regl

    # === REPORTING METHODS ===
    def iter_sampling_from_ls(self, z_rep, decode_attempts, num_iter,
                              noise_norm, constant_norm, verbose=False):
        print("\n### Searching molecules randomly sampled from " +
              "{:.2f} std (z-distance) from the point for {} iterations:" \
              .format(noise_norm, num_iter))
        dfs = []
        for i in tqdm(range(num_iter)):
            df = self.sample_and_dec_to_smiles(z_rep,
                                               decode_attempts=decode_attempts,
                                               noise_norm=noise_norm,
                                               constant_norm=constant_norm,
                                               verbose=verbose)
            if not len(df) > 0:
                continue
            else:
                dfs.append(df)

        out_df = pd.concat(dfs, ignore_index=True)
        if not len(out_df) > 0:
            print("Nothing was found after {:d}" + \
                   "iterations!".format(num_iter))
        else:
            out_df.drop_duplicates(subset='smiles', inplace=True)
            out_df.sort_values('distance', inplace=True)
            print("\n### Found {:d} unique and verified molecules!"
                  .format(len(out_df)))

        return out_df

    def save_gen_mols(self, df, cols_of_interest, out_file, out_dir):
        out_file = out_file.split(".")[0]
        alldf2 = df.sort_values(by=cols_of_interest,
                                ascending=[False] * len(cols_of_interest))
        alldf2.to_csv(out_dir + out_file + ".csv", index=False)
        alldf3 = alldf2.drop(['mol'], axis=1)
        alldf3.to_csv(out_dir + out_file + "_no_formula.csv", index=False)

        print("### Creating pdf file of generated molecules sorted by" + \
               " the given property...", end='')
        labels = ['SMILES: {:s}'.format(ss) for [ss] in
                  zip(alldf2['smiles'])]
        batch = 4  # number of molecules printed in each page of pdf file
        pdf = FPDF("L", "in", "Letter")
        count = 0
        for chunk in self.chunks(list(range(len(alldf2))), batch):
            tdf = alldf2.iloc[chunk, :]
            chunk_labels = [labels[i] for i in chunk]
            count = count + 1
            img = Draw.MolsToGridImage(mols=tdf['mol'], molsPerRow=2,
                                       subImgSize=(1000, 700),
                                       legends=chunk_labels,
                                       returnPNG=False)
            img.save(out_dir + "gen" + str(count) + ".png")
            pdf.add_page()
            pdf.image(out_dir + "gen" + str(count) + ".png", w=10, h=7)

        pdf.output(out_dir+out_file+".pdf", "F")
        listf = glob.glob(out_dir+"*.png")
        print("done!")
        for ff in listf:
            os.remove(ff)
        print('### ' + out_file + ".pdf for images of generated molecules" + \
              " is created successfully! Related CSV files are generated" + \
              "as well. They are all in " + out_dir + "\n")
        return

    def train_set_ls_sampler_w_prop(self, size=None, batch=2500,
                                    return_smiles=False):
        if self.data is None:
            print('use this sampler only for external property files')
            return

        cols = []
        if 'reg_prop_tasks' in self.params:
            cols += self.params['reg_prop_tasks']
        if 'logit_prop_tasks' in self.params:
            cols += self.params['logit_prop_tasks']
        idxs = self._random_idxs(size)
        smiles = [self.smiles[idx] for idx in idxs]
        data = [self.data.iloc[idx] for idx in idxs]
        Z = np.zeros((len(smiles), self.params['hidden_dim']))

        for chunk in self.chunks(list(range(len(smiles))), batch):
            sub_smiles = [smiles[i] for i in chunk]
            one_hot = self._smiles_to_hot(sub_smiles, self.params['SELFIES'])
            Z[chunk, :] = self.encode(one_hot)

        if return_smiles:
            return Z, data, smiles

        return Z, data

    def component_parity_check(self, input_data_name, ssize, seed,
                               parplot=False, histplot=False,
                               xlims=None, ylims=None, hexbinp=True):
        input_data = pd.read_csv(input_data_name)
        props = self.params['reg_prop_tasks']
        cols_of_interest = ['smiles'] + props
        #cols_of_interest = cols_.append(component)
        sub_data = input_data[cols_of_interest].sample(n=ssize,
                                                       random_state=seed)
        pred_prop_all = self.smiles_to_prop(sub_data.smiles)
        rmses = []

        for i, component in enumerate(props):
            all_obs = input_data[component]
            obs_comp = sub_data[component]
            pred_comp = pred_prop_all[:,i]

            if parplot:
                if hexbinp:
                    plt.hexbin(obs_comp, pred_comp, gridsize = 15, cmap ='Blues')
                else:
                    plt.scatter(obs_comp, pred_comp, s=30)
                plt.plot(obs_comp,obs_comp,linewidth=1.25, color='red')
                plt.title("Parity plot for " + component)
                plt.xlabel("Observed " + component)
                plt.ylabel("Predicted " + component)
                ax = plt.gca()
                if xlims is not None:
                    ax.set_xlim(xlims)
                if ylims is not None:
                    ax.set_ylim(ylims)
                plt.savefig(component + "_parity_plot.pdf", dpi=300,
                            bbox_inches='tight')
                plt.close()
                sys.stdout.write("Parity plot for " + component +
                                 " created successfully!\n")
                sys.stdout.flush()

            if histplot:
                # Histograms on top of each other
                r1 = np.mean(all_obs) - np.std(all_obs)
                r2 = np.mean(all_obs) + np.std(all_obs)
                bins = np.linspace(r1, r2, 100)
                plt.hist(obs_comp, bins, alpha=0.5, label='Observed')
                plt.hist(pred_comp, bins, alpha=0.5, label='Predicted')
                plt.legend(loc='upper right')
                plt.title("Histogram of " + component)
                plt.xlabel(component)
                #plt.axvline(np.mean(obs_comp), color='k', linestyle='dashed')
                plt.savefig(component + "_hist.pdf", dpi=300,
                            bbox_inches='tight')
                plt.close()
                sys.stdout.write("Histogram plot for " + component +
                                 " created successfully!\n")
                sys.stdout.flush()

            rmses.append(np.sqrt(np.mean((obs_comp-pred_comp)**2)))

        return rmses
