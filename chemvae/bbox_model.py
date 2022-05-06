
from abc import ABCMeta, abstractmethod


class BBoxModel(metaclass=ABCMeta):
    # === STANDARD CONSTRUCTOR ===
    def __init__(self):
        pass

    # === AUXILIARY METHODS ===
    # NOTE: Protected or Private methods here

    # === MANIPULATION METHODS ===
    # NOTE: Methods for training/modifying or otherwise altering the
    #       state of the model here.

    # === PROPERTY EVALUATION METHODS ===
    @abstractmethod
    def smiles_to_z(self, smiles):
        """Converts an array of SMILES strings to an array of latent space.

        Passes molecules through an encoder to get points in the latent space.

        Args:
            smiles: A numpy ndarray of strings.

        Returns:
            A numpy ndarray with the first dimension corresponding to the
            smiles first dimension, and the second dimension corresponding
            to the ls_dim.
        """
        return None

    @abstractmethod
    def z_to_smiles(self, z, verified=True):
        """Converts an array of latent points to an array of SMILES strings.

        Passes the latent space points through a decoder to get molecules.

        Args:
            z: A numpy ndarray of latent space points. First dimension
                corresponds to number of molecules, second dimension
                corresponds to size of the latent space (ls_dim).

        Returns:
            A list with a length equal to the number of molecules whose latent
            space representations passed to the function. The elements of the
            list are corresponding SMILES string, if verified. Otherwise, it
            will be 'None' showing that the decoded string was not a verified
            SMILES string.
        """
        return None

    @abstractmethod
    def predict_prop_z(self, z):
        """Converts an array of latent points to an array of properties.

        Passes the latent space points through a property prediction model
        to get properties.

        Args:
            z: A numpy ndarray of latent space points. First dimension
                corresponds to number of points to evaluate, second dimension
                corresponds to size of the latent space (ls_dim).

        Returns:
            A numpy ndarray with the first dimension corresponding to the
            number of points in the latent space points evaluated, and the
            second dimension corresponding to the number of properties
            predicted (n_props).
        """
        return None

    # === BASIC QUERY METHODS ===
    # NOTE: Simple methods for returning attributes of the class
    @property
    @abstractmethod
    def ls_dim(self):
        """Number of dimensions (i.e., size) of the latent space."""
        return None

    @property
    @abstractmethod
    def n_props(self):
        """Number of properties to be predicted."""
        return None

    # === REPORTING METHODS ===
    # NOTE: Methods for outputting to file or convenience methods
    #       for producing reports
