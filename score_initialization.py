import numpy as np
class score_initialization(object):
    """Fix scores initialization for each sub-array that can be accessible to use in parallel on the GPU."""
    def _score_list(self, ze, z_k, z_d, z_v):
        """Check if an entry gets assigned with +1 more than once.
        Args:
            ze (list): Digitized scores for one sample.
            z_k (list): Keep track if a value has occurred (keeps track on the position as well).
            z_d (dict): Bookkeeps the number of values has occurred.
            z_v (list): Keep track of the number of times a value has occurred (keeps track on the position as well).
        Returns:
             Lists that keep track of when and the number of time a value comes from the digitized array.
        """
        for z_item in ze:
            if not z_item in z_k:
                z_d[z_item] = 1
                z_v.append(1)
            else:
                z_d[z_item] += 1
                z_v.append(z_d[z_item])
            z_k.append(z_item)
        return z_k, z_d, z_v

    def _check_for_successor(self, z_k, i, v):
        """Check if an entry gets assigned with +1 more than once.
        Args:
            z_k (array): Digitized scores.
            i (int): Position of value in array.
            v (int): Value of that entry.
        Returns:
             The list that points to the possible ancestor(if a value occurs more than once).
        """
        F = list()
        for I, vx in enumerate(z_k[i + 1:]):
            if v == vx:
                F.append(i + I + 1)
        return F

    def _get_value_rank(self, z_k):
        """Keep track of when an entry is added with one additional +1. Important for the GPU.
        Args:
            z_k (array): Digitized scores.
        Returns:
            Return a list that points to the ancestor of one particular entry occurs multiple times(if they occur more than once) for each score list when an). 
        """
        rank = list()
        for i, v in enumerate(z_k):
            F = self._check_for_successor(z_k, i, v)
            if len(F)>0:
                rank.append(min(F))
            else:
                #Cardinal (only occours once)
                rank.append(10**6)
        return rank

    def _score_lists(self, digitized_score):
        """Convert score lists (one for each sample) to one complete array.
        Args:
            digitized_score (array): Digitized scores.
        Returns:
             The list that contains initialized score lists, one score list for each sample.
             The length of longest list.
        """
        Z = list()
        L = 0
        for ze in digitized_score:
            z_k, z_v, z_d = list(), list(), dict()
            z_k, z_d, z_v = self._score_list(ze, z_k, z_d, z_v)
            rank = self._get_value_rank(z_k)

            if len(z_k) > L:
                L = len(z_k)
            Z.append([z_k,z_v,rank])

        return Z, L

    def score_to_array(self, Z, L):
        """Convert score lists (one for each sample) to one complete array. 
        Args:
            Z (list(list)): List with initial score lists.
            L (int): Length of the longest initial score list.
        Returns:
            The initialized score array for each fresh new sub-array.
        """
        z = np.zeros([len(Z), 3, L])
        for i, z_s in enumerate(Z):
            for k, z_e in enumerate(z_s):
                z[i,k,0:len(z_e)] = np.asarray(z_e)
        return z

    def get_score_init(self, digitized_score):
        """Get initial score array. Keep track on which entry that should be assigned with a +1 and the number of times(by keeping track on ascendency), for all samples.
        Args:
            digitized_score (array): Digitized scores.
        Returns:
            The initialized score array for each fresh new sub-array.
        """
        Z, L  = self._score_lists(digitized_score)
        return self.score_to_array(Z, L)

""" def _score_list(self, ze, z_k, z_d, z_v):
        for i, z_item in enumerate(ze):
            if not z_item in z_k:
                z_k[i] = z_item
                z_d[z_item] = 1
                z_v[i] = 1
            else:
                
                z_k[i] = z_item
                z_d[z_item] += 1
                z_v[z_k == z_item] = z_d[z_item]
        return z_k.tolist(), z_d, z_v.tolist() """