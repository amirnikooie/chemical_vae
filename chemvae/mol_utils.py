import pandas as pd
import numpy as np
import pickle as pkl
from rdkit.Chem import AllChem as Chem
import logging
import selfies as sf
import sys

logging.getLogger('autoencoder')
logging.getLogger().setLevel(20)
logging.getLogger().addHandler(logging.StreamHandler())


# =================
# text io functions
# ==================

###=== Related to Selfies and its comparison with SMILES
def smile_to_hot(smile, largest_smile_len, alphabet):
    """Go from a single smile string to a one-hot encoding.
    """

    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with ' '
    smile += ' ' * (largest_smile_len - len(smile))

    # integer encode input smile
    integer_encoded = [char_to_int[char] for char in smile]

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)


def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    """Convert a list of smile strings to a one-hot encoding

    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """

    hot_list = []
    for s in smiles_list:
        _, onehot_encoded = smile_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)


def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """Go from a single selfies string to a one-hot encoding.
    """

    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)


def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to a one-hot encoding
    """

    hot_list = []
    for s in selfies_list:
        _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)

def mysf_encoder(smi):
    try:
        return sf.encoder(smi)
    except:
        return None

def get_selfie_and_smiles_encodings_for_dataset(smiles_list):
    """
    Returns encoding, alphabet and length of largest molecule in SMILES and
    SELFIES, given a file containing SMILES molecules.
    input:
        csv file with molecules. Column's name must be 'smiles'.
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """

    #df = pd.read_csv(file_path)

    smiles_list = np.asanyarray(smiles_list)

    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding

    largest_smiles_len = len(max(smiles_list, key=len))

    #print('--> Translating SMILES to SELFIES...')
    # Filtering those that violate semantic constraints
    selfies_list = np.array(list(map(mysf_encoder, smiles_list)))
    selfies_list = list(selfies_list[selfies_list!=np.array(None)])

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]')
    selfies_alphabet = list(all_selfies_symbols)

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    #print('Finished translating SMILES to SELFIES.')

    return selfies_list, selfies_alphabet, largest_selfies_len, \
           smiles_list, smiles_alphabet, largest_smiles_len

###=======================

#==== Main chem_vae code for this file

def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        pass
    return None


def verify_smiles(smile):
    return (smile != '') and pd.notnull(smile) and (Chem.MolFromSmiles(smile) is not None)


def good_smiles(smile):
    if verify_smiles(smile):
        return canon_smiles(smile)
    else:
        return None


def pad_smile(string, max_len, padding='right'):
    if len(string) <= max_len:
        if padding == 'right':
            return string + " " * (max_len - len(string))
        elif padding == 'left':
            return " " * (max_len - len(string)) + string
        elif padding == 'none':
            return string


def filter_valid_length(strings, max_len):
    return [s for s in strings if len(s) <= max_len]


def filter_valid_smiles_return_invalid(strings, max_len):
    filter_list = []
    new_smiles = []
    for idx, s in enumerate(strings):
        if len(s) > max_len:
            filter_list.append(idx)
        else:
            new_smiles.append(s)
    return new_smiles, filter_list


def smiles_to_hot(smiles, max_len, padding, char_indices, nchars):
    smiles = [pad_smile(i, max_len, padding)
              for i in smiles if pad_smile(i, max_len, padding)]

    X = np.zeros((len(smiles), max_len, nchars), dtype=np.float32)

    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            try:
                X[i, t, char_indices[char]] = 1
            except KeyError as e:
                print("ERROR: Check chars file. Bad SMILES:", smile)
                raise e
    return X


def smiles_to_hot_filter(smiles, char_indices):
    filtered_smiles = []
    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            try:
                char_indices[char]
            except KeyError:
                break
        else:
            filtered_smiles.append(smile)
    return filtered_smiles


def term_hot_to_smiles(x, temperature, indices_chars):
    temp_string = ""
    for j in x:
        index = thermal_argmax(j, temperature)
        temp_string += indices_chars[index]
    return temp_string


def hot_to_smiles(hot_x, alphabet, selfies): ## Changed to aplhabet. Work here...
    smiles = []
    for x in hot_x:
        temp_str = ''
        for j in x:
            index = np.argmax(j)
            temp_str += alphabet[index]
        temp_str = temp_str.replace(' ', '')
        if selfies:  # if SELFIES, decode to SMILES
            temp_str = sf.decoder(temp_str)
        smiles.append(temp_str)
    return smiles


def thermal_argmax(prob_arr, temperature):
    prob_arr = np.log(prob_arr) / temperature
    prob_arr = np.exp(prob_arr) / np.sum(np.exp(prob_arr))
    print(prob_arr)
    if np.greater_equal(prob_arr.sum(), 1.0000000001):
        logging.warn('Probabilities to sample add to more than 1, {}'.
                     format(prob_arr.sum()))
        prob_arr = prob_arr / (prob_arr.sum() + .0000000001)
    if np.greater_equal(prob_arr.sum(), 1.0000000001):
        logging.warn('Probabilities to sample still add to more than 1')
    return np.argmax(np.random.multinomial(1, prob_arr, 1))


def load_smiles(smi_file, max_len=None, return_filtered=False):
    if smi_file[-4:] == '.pkl':
        with open(smi_file, 'rb') as f:
            smiles = pkl.load(f)
    else:  # assume file is a text file
        with open(smi_file, 'r') as f:
            smiles = f.readlines()
        smiles = [i.strip() for i in smiles]

    if max_len is not None:
        if return_filtered:
            smiles, filtrate = filter_valid_smiles_return_invalid(
                smiles, max_len)
            if len(filtrate) > 0:
                print('Filtered {} smiles due to length'.format(len(filtrate)))
            return smiles, filtrate

        else:
            old_len = len(smiles)
            smiles = filter_valid_length(smiles, max_len)
            diff_len = old_len - len(smiles)
            if diff_len != 0:
                print('Filtered {} smiles due to length'.format(diff_len))

    return smiles


def load_smiles_and_data_df(data_file, max_len, reg_tasks=None, logit_tasks=None, normalize_out=None, dtype='float64'):
    # reg_tasks : list of columns in df that correspond to regression tasks
    # logit_tasks : list of columns in df that correspond to logit tasks

    if logit_tasks is None:
        logit_tasks = []
    if reg_tasks is None:
        reg_tasks = []
    df = pd.read_csv(data_file)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip()
    df = df[df.iloc[:, 0].str.len() <= max_len]
    smiles = df.iloc[:, 0].tolist()

    reg_data_df = df[reg_tasks]
    logit_data_df = df[logit_tasks]
    # Load regression tasks
    if len(reg_tasks) != 0 and normalize_out is not None:
        df_norm = pd.DataFrame(reg_data_df.mean(axis=0), columns=['mean'])
        df_norm['std'] = reg_data_df.std(axis=0)
        reg_data_df = (reg_data_df - df_norm['mean']) / df_norm['std']
        df_norm.to_csv(normalize_out)

    if len(logit_tasks) != 0 and len(reg_tasks) != 0:
        return smiles, np.vstack(reg_data_df.values).astype(dtype), np.vstack(logit_data_df.values).astype(dtype)
    elif len(reg_tasks) != 0:
        return smiles, np.vstack(reg_data_df.values).astype(dtype)
    elif len(logit_tasks) != 0:
        return smiles, np.vstack(logit_data_df.values).astype(dtype)
    else:
        return smiles


def smiles2one_hot_chars(smi_list, max_len):
    # get all the characters
    char_lists = [list(smi) for smi in smi_list]
    chars = list(set([char for sub_list in char_lists for char in sub_list]))
    chars.append(' ')

    return chars


def make_charset(smi_file, char_file):
    with open(smi_file, 'r') as afile:
        unique_chars = set(afile.read())
    bad = ['\n', '"']
    unique_chars = [c for c in unique_chars if c not in bad]
    unique_chars.append(' ')
    print('found {} unique chars'.format(len(unique_chars)))
    astr = str(unique_chars).replace("\'", "\"")
    print(astr)

    with open(char_file, 'w') as afile:
        afile.write(astr)

    return


# =================
# data parsing io functions
# ==================

def CheckSmiFeasible(smi):
    # See if you can make a smiles with mol object
    #    if you can't, then skip
    try:
        get_molecule_smi(Chem.MolFromSmiles(smi))
    except:
        return False
    return True


def balanced_parentheses(input_string):
    s = []
    balanced = True
    index = 0
    while index < len(input_string) and balanced:
        token = input_string[index]
        if token == "(":
            s.append(token)
        elif token == ")":
            if len(s) == 0:
                balanced = False
            else:
                s.pop()

        index += 1

    return balanced and len(s) == 0


def matched_ring(s):
    return s.count('1') % 2 == 0 and s.count('2') % 2 == 0


def fast_verify(s):
    return matched_ring(s) and balanced_parentheses(s)


def get_molecule_smi(mol_obj):
    return Chem.MolToSmiles(mol_obj)


def canon_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)


if __name__ == '__main__':
    # print("please import me")
    smiles, reg_dat, logit_dat = load_smiles_and_data_df("zinc/250k_rndm_zinc_drugs_clean_5.csv", 120,
                                                         ['logP', 'qed', 'SAS'], ['NRingsGT6', 'PAINS'])
    print(smiles[:5])
    print(reg_dat[:5, :])
    print(logit_dat[:5, :])
