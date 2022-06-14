from vae_model import *

#======================
""" Creating the VAE object and initializing the instance """
model_DIR = "./aux_files/"

vae = VAE_Model(directory=model_DIR)
print("The VAE object created successfully!")

#============
""" Working with a sample smiles string to reconstruct it and predict its
    properties """

sample_smiles = 'OC1=CC=C(C2=C(C3=CC=C(O)C=C3S2)N2C3=CC=C(C=C3C=C2)OCCN2CCCCC2)C=C1'
z_rep = vae.smiles_to_z(sample_smiles)
X_hat = vae.z_to_smiles(z_rep, verified=False) # decoding
        # to molecular space without verifying its validity.
predicted_props = vae.predict_prop_z(z_rep)

print("### {:20s} : {}".format('Input', sample_smiles))
print("### {:20s} : {}".format('Reconstruction', X_hat[0]))
print("### {:20s} : {} with norm {:.3f}".format('Z representation:',
                                                z_rep.shape,
                                                np.linalg.norm(z_rep)))
print("### {:20s} : {}".format('Number of properties', vae.n_props))
print("### {:20s} : {}\n\n".format('Predicted properties', predicted_props))

#======================
""" Property prediction for 20 samples from multivariate standard normal
    distribution """
z_mat = np.random.normal(0, 1, size=(20,z_rep.shape[1]))
pred_prop = vae.predict_prop_z(z_mat)

#======================
""" Converting those random representations to valid molecules """
x_hat_list = vae.z_to_smiles(z_mat, verified=True) # decoding
            # to valid molecules
verified_x_hat = [item for item in x_hat_list if item!='None']
print("\n### {} out of 20 compounds are verified!".format(len(verified_x_hat)))
print("### {:20s} : {}".format('Predicted properties:', pred_prop))


#======================
""" Iteratively sampling from vicinity of a point in the latent space """
df = vae.iter_sampling_from_ls(z_rep, decode_attempts=500, num_iter=10,
                               noise_norm=0.5, constant_norm=False,
                               verbose=False)
#======================
""" Saving generated molecules to a pdf file as well as CSV using data
    frame above """
vae.save_gen_mols(df, cols_of_interest=['comp1','comp2','comp3'],
                  out_file="gen_mols.pdf", out_dir="./test_out/")

#======================
""" prediction performance analysis for a component of interest with option of
    drawing parity plot for that component """


#input_data = string showing the location and name of the dataset to for
#              prediction perfomance analysis. Could be the test set.

filename = 'small_train.csv'
nsamples = 4000 #600000

rmses = vae.component_parity_check(model_DIR+filename, ssize=nsamples, seed=435,
                                 histplot=True, parplot=True, hexbinp=False)
                                 #xlims=[0,1], ylims=[0,1])

print(rmses)
