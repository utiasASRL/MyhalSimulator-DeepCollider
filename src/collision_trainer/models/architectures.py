#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

from models.blocks import KPConv, UnaryBlock, block_decider, LRFBlock, GlobalAverageBlock, \
    ProjectorBlock, Propagation2DBlock, Initial2DBlock
import numpy as np
import time
import torch
import torch.nn as nn

from utils.mayavi_visu import fast_save_future_anim, save_zoom_img


def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


def rot_loss(net, rotated_features, rots, all_lengths, config, rand_n=10, verbose=False):

    # Init
    dim = int(rots[0].shape[0])
    layer_N = np.array([int(torch.sum(lengths)) for lengths in all_lengths])
    l2_loss = torch.nn.MSELoss(reduction='mean')

    # Get random inds and corresponding rotations for each layer
    all_rots = []
    all_elem_inds = []
    for l_i, l_N in enumerate(layer_N):
        l_N = int(l_N)
        if rand_n < l_N:
            random_inds = torch.randperm(l_N)[:rand_n]
        else:
            random_inds = torch.arange(l_N)
        i0 = 0
        layer_elem_inds = []
        layer_rots = []
        for b_i, length in enumerate(all_lengths[l_i]):
            elem_inds = random_inds[(random_inds >= i0) & (random_inds < i0 + length)]
            layer_elem_inds.append(elem_inds)
            layer_rots.append((rots[b_i].unsqueeze(0)).expand((len(elem_inds), -1, -1)))
            i0 += length
        all_elem_inds.append(torch.cat(layer_elem_inds, dim=0))
        all_rots.append(torch.cat(layer_rots, dim=0))

    invar_loss = 0
    equivar_loss = 0
    for i, x in enumerate(net.intermediate_features):

        # Get features from rotated input
        rot_x = rotated_features[i].detach()

        # Get layer index of these features
        l_i = int(np.where(layer_N == int(x.shape[0]))[0])

        # Get number of invariant/equivariant features
        N_f = int(x.shape[1])
        N_inv = int(np.floor(float(x.shape[1]) * config.invar_ratio))
        N_equi = int(np.floor(float(x.shape[1]) * config.equivar_ratio)) // dim
        N_equi = min(N_equi, N_f // dim)
        fi1 = N_equi * dim
        fi2 = min(fi1 + N_inv, N_f)

        # Compute equivariance loss
        if fi1 > 0:

            # Realign the equivariant features
            elem_inds = all_elem_inds[l_i]
            equi_x = torch.reshape(rot_x[elem_inds, :fi1], (len(elem_inds), N_equi, 3))
            v1 = torch.matmul(equi_x, all_rots[l_i].transpose(1, 2))
            v2 = torch.reshape(x[elem_inds, :fi1], (len(elem_inds), N_equi, 3))
            sq_diff = torch.sum((v1 - v2) ** 2, dim=2)
            if config.equivar_oriented:
                equivar_loss += torch.mean(sq_diff)
            else:
                sq_diff_opposite = torch.sum((v1 + v2) ** 2, dim=2)
                equivar_loss += torch.mean(torch.min(sq_diff_opposite, sq_diff))

            if verbose:
                dot_prod = np.abs(np.sum(v1 * v2, axis=1))
                norm_prod = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
                cosine_errors = dot_prod / (norm_prod + 1e-6)
                AE = np.arccos(np.clip(cosine_errors, -1.0, 1.0))
                fmt_str = 'Feature {:2d} AE: min={:.1f}, mean={:.1f}, max={:.1f}'
                print(fmt_str.format(i, np.min(AE) * 180 / np.i,
                                     np.mean(AE) * 180 / np.i,
                                     np.max(AE) * 180 / np.i))

        # Compute invariance loss
        if fi1 < N_f and fi2 > fi1:

            # Apply invar_loss
            invar_loss += l2_loss(x[all_elem_inds[l_i], fi1:fi2], rot_x[all_elem_inds[l_i], fi1:fi2])
            print(x.shape, float(x[0, 0]), float(rot_x[0, 0]), invar_loss)

            a = 1/0

    # Average over the number of blocks
    return invar_loss / len(net.intermediate_features), equivar_loss / len(net.intermediate_features)


def orthogonalization_loss(net):

    l2_loss = torch.nn.MSELoss(reduction='mean')
    ortho_loss = 0

    for m in net.modules():

        if isinstance(m, LRFBlock):

            pdim = int(m.pred_rots.shape[-1])

            # Reshape predicted local reference frames
            pred_rots = m.pred_rots.reshape((-1, pdim, pdim))

            # Get the matrix "ratio" R x R^T
            mat_diff = torch.matmul(pred_rots, pred_rots.detach().transpose(1, 2))

            # Corresponding target is identity
            target = torch.eye(pdim, dtype=torch.float32, device=mat_diff.device).unsqueeze(0)

            # Compute loss
            ortho_loss += l2_loss(mat_diff, target)

    #print(ortho_loss * net.ortho_power)

    return ortho_loss * net.ortho_power


class KPCNN(nn.Module):
    """
    Class defining KPCNN
    """

    def __init__(self, config):
        super(KPCNN, self).__init__()

        ####################
        # Network operations
        ####################

        # Current radius of convolution and feature dimension
        self.architecture = config.architecture
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        in_lrf = 1
        up_lrf = config.first_n_lrf

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            # if ('equivariant' in block) and (not out_dim % 3 == 0):
            #     raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function
            if 'equivariant' in block:
                self.block_ops.append(block_decider(block, r, in_dim, out_dim, layer, config,
                                                    n_lrf=in_lrf, up_lrf=up_lrf))
            else:
                self.block_ops.append(block_decider(block, r, in_dim, out_dim, layer, config))

            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0
                in_lrf = in_lrf * up_lrf
                up_lrf = config.lrf_up_factor

            else:
                in_lrf = in_lrf * up_lrf
                up_lrf = 1

        if 'equivariant' in config.architecture[-1]:
            self.head_mlp = UnaryBlock(out_dim * 2, 1024, False, 0)
            self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0)
        else:
            self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
            self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0)

        ################
        # Network Losses
        ################

        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.ortho_power = config.ortho_power
        self.l1 = nn.L1Loss()
        self.intermediate_features = []
        self.debug_lrf = []

        return

    def forward(self, batch, config, save_block_features=False, debug_lrf=False):

        # Init intermediate features container
        self.intermediate_features = []
        self.debug_lrf = []

        # Save all block operations in a list of modules
        x = batch.features.clone().detach()
        lrf = batch.lrf.clone().detach()

        # Loop over consecutive blocks
        for block_op in self.block_ops:

            # Apply the block
            if ('equivariant' in block_op.block_name):
                x, lrf = block_op(x, batch, lrf)
                if debug_lrf and lrf is not None:
                    self.debug_lrf.append(lrf)

            else:
                x = block_op(x, batch)

            # Optionally save features
            if save_block_features and type(block_op) != GlobalAverageBlock:
                self.intermediate_features.append(x)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, batch):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param batch: batch struct containing labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, batch.labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Regularzation for equivariant convs
        if np.any(['equivariant' in layer_name for layer_name in self.architecture]):
            self.reg_loss += orthogonalization_loss(self)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, batch):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = batch.labels.size(0)
        correct = (predicted == batch.labels).sum().item()

        return correct / total


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls, num_parts=None):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        in_lrf = 4
        up_lrf = config.first_n_lrf

        # Special case if multi part segmentation
        self.num_parts = num_parts

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            # if ('equivariant' in block) and (not out_dim % 3 == 0):
            #     raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function
            if 'equivariant' in block:
                self.encoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config,
                                                         n_lrf=in_lrf, up_lrf=up_lrf))
            else:
                self.encoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                in_lrf = in_lrf * up_lrf
                up_lrf = config.lrf_up_factor

            else:
                in_lrf = in_lrf * up_lrf
                up_lrf = 1

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function
            self.decoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))


            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        if num_parts is None:
            self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
            self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)
        else:
            maxC = np.max(num_parts)
            head_dim = min(config.first_features_dim, len(num_parts) * maxC * 2)
            self.head_mlp = UnaryBlock(out_dim, head_dim, False, 0)
            self.head_softmax = UnaryBlock(head_dim, len(num_parts) * maxC, False, 0)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        if num_parts is None:
            self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])
        else:
            maxC = np.max(num_parts)
            self.valid_labels = np.arange(maxC)

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()
        self.intermediate_features = []

        return

    def forward(self, batch, config, save_block_features=False):

        # Init intermediate features container
        self.intermediate_features = []

        # Get input features
        x = batch.features.clone().detach()

        if np.any(['equivariant' in layer_name for layer_name in config.architecture]):
            lrf = batch.lrf.clone().detach()
        else:
            lrf = None

        #################
        # Encoder network
        #################

        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):

            # Get skip feature if necessary
            if block_i in self.encoder_skips:
                skip_x.append(x)

            # Apply the block
            if ('equivariant' in block_op.block_name):
                x, lrf = block_op(x, batch, lrf)
            else:
                x = block_op(x, batch)

            # Optionally save features
            if save_block_features:
                self.intermediate_features.append(x)

        #################
        # Decoder Network
        #################

        for block_i, block_op in enumerate(self.decoder_blocks):

            # Concat with skip
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)

            # Apply block
            x = block_op(x, batch)

            # Optionally save features
            if save_block_features:
                self.intermediate_features.append(x)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        # Special case of multi-head
        if self.num_parts is not None:

            # Reshape to get each part outputs
            maxC = np.max(self.num_parts)
            x = x.reshape((-1, len(self.num_parts), maxC))

            # Collect the right logits for the object label
            new_x = []
            i0 = 0
            for b_i, length in enumerate(batch.lengths[0]):
                new_x.append(x[i0:i0 + length, batch.obj_labels[b_i], :])
                i0 += length
            x = torch.cat(new_x, dim=0)

        return x

    def loss(self, outputs, batch):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param batch: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(batch.labels)
        for i, c in enumerate(self.valid_labels):
            target[batch.labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.squeeze().unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, batch):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(batch.labels)
        for i, c in enumerate(self.valid_labels):
            target[batch.labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total


class KPFCNN_regress(nn.Module):
    """
    Class defining KPFCNN model used for regression tasks
    """

    def __init__(self, config):
        super(KPFCNN_regress, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.regress_dim = config.num_classes

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_regress = UnaryBlock(config.first_features_dim, self.regress_dim, False, 0)

        ################
        # Network Losses
        ################

        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()
        self.cosine_similarity = nn.CosineSimilarity(eps=1e-6)
        self.pdist = nn.PairwiseDistance()
        self.intermediate_features = []

        return

    def forward(self, batch, config, save_block_features=False):

        # Init intermediate features container
        self.intermediate_features = []

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
            if save_block_features:
                self.intermediate_features.append(x)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
            if save_block_features:
                self.intermediate_features.append(x)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_regress(x, batch)

        return x

    def loss(self, outputs, batch, cosine=False, oriented=False, harder=0.7):
        """
        Runs the loss on outputs of the model
        :param outputs: [N, 3] predicted normals
        :param batch: batch containing the [N, 3] groundtruth normals
        :return: loss
        """

        targets = batch.normals

        # Get the distance to minimize
        if cosine:
            cos_sim = self.cosine_similarity(outputs, targets)
            if oriented:
                dist = -cos_sim + 1
            else:
                dist = -torch.abs(cos_sim) + 1
        else:
            if oriented:
                dist = torch.sum((outputs - targets) ** 2, dim=1)
            else:
                a = torch.sum((outputs - targets) ** 2, dim=1)
                b = torch.sum((outputs + targets) ** 2, dim=1)
                dist = torch.min(a, b)

        # Train on the 30% hardest elements
        if harder > 0:
            dist_detach = dist.cpu().detach()
            percent = int(np.floor(100 * harder))
            hard_limit = np.percentile(dist_detach, percent)
            dist = dist[dist_detach > hard_limit]

        # Loss is averaged over all points
        self.output_loss = torch.mean(dist)

        # Regularization of deformable offsets
        self.reg_loss = 0
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss += p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, batch):
        return self.output_loss


class KPRCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total


class KPCollider(nn.Module):
    """
    Class defining KPFCollider
    """

    def __init__(self, config, lbl_values, ign_lbls, num_parts=None):
        super(KPCollider, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)
        in_lrf = 4
        up_lrf = config.first_n_lrf

        # Special case if multi part segmentation
        self.num_parts = num_parts

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            # if ('equivariant' in block) and (not out_dim % 3 == 0):
            #     raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function
            if 'equivariant' in block:
                self.encoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config,
                                                         n_lrf=in_lrf, up_lrf=up_lrf))
            else:
                self.encoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                in_lrf = in_lrf * up_lrf
                up_lrf = config.lrf_up_factor

            else:
                in_lrf = in_lrf * up_lrf
                up_lrf = 1

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function
            self.decoder_blocks.append(block_decider(block, r, in_dim, out_dim, layer, config))


            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        if num_parts is None:
            self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
            self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)
        else:
            maxC = np.max(num_parts)
            head_dim = min(config.first_features_dim, len(num_parts) * maxC * 2)
            self.head_mlp = UnaryBlock(out_dim, head_dim, False, 0)
            self.head_softmax = UnaryBlock(head_dim, len(num_parts) * maxC, False, 0)


        ########################
        # 2D Propagation network
        ########################

        # Project vertically (average pooling) only in the inscribed square c= sqrt(2)*r
        # Use precomputed projection indices (same as for our point pooling) limit to something like 10 indexes
        # and chose randomly (this would play the same role as a dropout)
        self.projector = ProjectorBlock(config.detach_2D, pooling='max')

        # First conv out ou projection to propagate feature a first time
        self.initial_net = Initial2DBlock(out_dim, config.first_features_dim)
        self.init_softmax_2D = nn.Conv2d(config.first_features_dim, 3, kernel_size=1, bias=True)
        self.merge_softmax_2D = nn.Conv2d(config.first_features_dim, 3, kernel_size=1, bias=True)

        self.shared_2D = config.shared_2D
        if self.shared_2D:
            # Use a mini network for propagation, which is repeated at every step
            self.prop_net = Propagation2DBlock(config.first_features_dim, config.first_features_dim)

            # Shared head softmax
            self.head_softmax_2D = nn.Conv2d(config.first_features_dim, 3, kernel_size=1, bias=True)

        else:
            self.prop_net = nn.ModuleList()
            self.head_softmax_2D = nn.ModuleList()
            for i in range(config.n_2D_layers):
                self.prop_net.append(Propagation2DBlock(config.first_features_dim, config.first_features_dim))
                self.head_softmax_2D.append(nn.Conv2d(config.first_features_dim, 3, kernel_size=1, bias=True))


        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        if num_parts is None:
            self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])
        else:
            maxC = np.max(num_parts)
            self.valid_labels = np.arange(maxC)

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()
        self.intermediate_features = []

        # 2D parameters
        self.criterion_2D = torch.nn.BCEWithLogitsLoss()
        self.output_2D_loss = 0
        self.power_2D_init_loss = config.power_2D_init_loss
        self.power_2D_prop_loss = config.power_2D_prop_loss
        self.neg_pos_ratio = config.neg_pos_ratio
        self.train_3D = config.train_3D
        
        self.fixed_conv = torch.nn.Conv2d(config.n_2D_layers, config.n_2D_layers, 3, stride=1, padding=1, bias=False)
        self.fixed_conv.weight.requires_grad = False
        self.fixed_conv.weight *= 0
        for i in range(config.n_2D_layers):
            self.fixed_conv.weight[i, i] += 10000.0

        return

    def forward(self, batch, config, save_block_features=False):

        # Init intermediate features container
        self.intermediate_features = []

        # Get input features
        x = batch.features.clone().detach()

        if np.any(['equivariant' in layer_name for layer_name in config.architecture]):
            lrf = batch.lrf.clone().detach()
        else:
            lrf = None

        #################
        # Encoder network
        #################

        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):

            # Get skip feature if necessary
            if block_i in self.encoder_skips:
                skip_x.append(x)

            # Apply the block
            if ('equivariant' in block_op.block_name):
                x, lrf = block_op(x, batch, lrf)
            else:
                x = block_op(x, batch)

            # Optionally save features
            if save_block_features:
                self.intermediate_features.append(x)

        #################
        # Decoder Network
        #################

        for block_i, block_op in enumerate(self.decoder_blocks):

            # Concat with skip
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)

            # Apply block
            x = block_op(x, batch)

            # Optionally save features
            if save_block_features:
                self.intermediate_features.append(x)
                
        ############
        # 2D Network
        ############

        # Project feature from [N, D_0] to [B, D_0, L, L]
        x_2D = self.projector(x, batch)

        # Initial net
        x_2D = self.initial_net(x_2D)
        
        # Initial preds
        preds_2D = [self.init_softmax_2D(x_2D), self.merge_softmax_2D(x_2D)]

        # Append propagated preds
        if self.shared_2D:
            for i in range(config.n_2D_layers):
                x_2D = self.prop_net(x_2D)
                preds_2D.append(self.head_softmax_2D(x_2D))
        else:
            for i in range(config.n_2D_layers):
                x_2D = self.prop_net[i](x_2D)
                preds_2D.append(self.head_softmax_2D[i](x_2D))

        # Stack 2d outputs should have the shape: [B, 3, T_2D+2, L_2D, L_2D]
        preds_2D = torch.stack(preds_2D, axis=2)

        ##############
        # Head Network
        ##############
            
        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        # Special case of multi-head
        if self.num_parts is not None:

            # Reshape to get each part outputs
            maxC = np.max(self.num_parts)
            x = x.reshape((-1, len(self.num_parts), maxC))

            # Collect the right logits for the object label
            new_x = []
            i0 = 0
            for b_i, length in enumerate(batch.lengths[0]):
                new_x.append(x[i0:i0 + length, batch.obj_labels[b_i], :])
                i0 += length
            x = torch.cat(new_x, dim=0)

        return x, preds_2D.permute(0, 2, 3, 4, 1)

    def loss(self, outputs, batch):
        """
        Runs the loss on outputs of the model
        :param outputs: logits from 3D seg net and logits from 2D prop net
        :param batch: labels
        :return: loss
        """

        # Parse network outputs:
        outputs_3D = outputs[0]
        outputs_2D = outputs[1]

        #########
        # 3D Loss
        #########

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(batch.labels)
        for i, c in enumerate(self.valid_labels):
            target[batch.labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs_3D = torch.transpose(outputs_3D, 0, 1)
        outputs_3D = outputs_3D.unsqueeze(0)
        target = target.squeeze().unsqueeze(0)

        # Cross entropy loss
        if self.train_3D:
            self.output_3D_loss = self.criterion(outputs_3D, target)
        else:
            self.output_3D_loss = 0


        #########
        # 2D Loss
        #########

        # Binary cross entropy loss for multilable classification (because the labels are not mutually exclusive)
        # Only apply loss to part of the empyty space to reduce unbalanced classes

        # Init loss for initial class probablitities => shapes = [B, 1, L_2D, L_2D, 3]
        init_2D_loss = self.power_2D_init_loss * self.criterion_2D(outputs_2D[:, 0, :, :, :], batch.future_2D[:, 0, :, :, :])

        # Init loss for merged future class probablitities => shapes = [B, 1, L_2D, L_2D, 3]
        merged_future = batch.future_2D[:, 0, :, :, :].detach().clone()
        max_v, _ = torch.max(batch.future_2D[:, 1:, :, :, 2], dim=1)
        merged_future[:, :, :, 2] = max_v
        init_2D_loss += self.power_2D_init_loss * self.criterion_2D(outputs_2D[:, 1, :, :, :], merged_future)
        

        # Propagation loss applied to each prop layer => shapes = [B, T_2D, L_2D, L_2D, 3]
        # Only use the positive pixels and approx 2 times more negative pixels (picked randomly)
        future_gt = batch.future_2D[:, 1:, :, :, 2]
        ratio_pos = float(torch.sum((future_gt > 0.01).to(torch.float32)) / float(torch.numel(future_gt)))

        # use a fixed conv2d to dilate positive inds
        with torch.no_grad():
            dilated_gt = self.fixed_conv(future_gt)
        dilated_inds = dilated_gt > 0.01

        # Add some random negative inds
        future_p = outputs_2D[:, 2:, :, :, 2]
        loss_mask = torch.rand_like(future_p)
        loss_mask[dilated_inds] = 1.0
        threshold = min(1.0 - ratio_pos * self.neg_pos_ratio, 0.9)
        loss_mask = loss_mask > threshold
        prop_2D_loss = self.power_2D_prop_loss * self.criterion_2D(future_p[loss_mask], future_gt[loss_mask])

        # Sum the two 2D losses
        self.output_2D_loss = init_2D_loss + prop_2D_loss

        ######
        # Regu
        ######

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        self.output_loss = self.output_3D_loss + self.output_2D_loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, batch):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Parse network outputs:
        outputs_3D = outputs[0]
        outputs_2D = outputs[1]

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(batch.labels)
        for i, c in enumerate(self.valid_labels):
            target[batch.labels == c] = i

        predicted = torch.argmax(outputs_3D.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total


















