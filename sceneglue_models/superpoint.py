from pathlib import Path
import torch
from torch import nn

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors_new(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - 1), (h*s - 1)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor
    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 3,
        'keypoint_threshold': 0.0,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        for p in self.parameters():
            p.requires_grad = False

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)))
        print('Load official SuperPoint model')
        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        self.lin2 = nn.Conv1d(c3, c3, kernel_size=1)
        self.lin1 = nn.Conv1d(c2, c2, kernel_size=1)
        self.lin0 = nn.Conv1d(c1, c1, kernel_size=1)
        self.merge = nn.Conv1d(c5 + c3 + c2 + c1, c5, kernel_size=1)

    def add_random_point(self, keypoints, scores):
        for i, (k, s) in enumerate(zip(keypoints, scores)):
            """
            Occurence of below condition is very rare as we are sampling keypoints above threshold of 0 itself and then
            sampling max_keypoints from it. But incase if it happends then we are randomly adding some pixel locations to 
            without checking any conditions with respect to preexisting keypoints.
            """
            if len(k) < self.config['max_keypoints']:
                to_add_points = self.config['max_keypoints'] - len(k)
                random_keypoints = torch.stack(
                    [torch.randint(0, self.w * 8, (to_add_points,), dtype=torch.float32, device=k.device),
                     torch.randint(0, self.h * 8, (to_add_points,), dtype=torch.float32, device=k.device)], 1)
                keypoints[i] = torch.cat([keypoints[i], random_keypoints], dim=0)
                scores[i] = torch.cat(
                    [scores[i], torch.zeros(to_add_points, dtype=torch.float32, device=s.device) * 0.1], dim=0)

    def forward(self, data, mode='test', max_kpt=None):
        """ Compute keypoints, scores, descriptors for image """
        max_keypoints = max_kpt if max_kpt is not None else self.config['max_keypoints']
        # Shared Encoder
        x0 = self.relu(self.conv1a(data['image0']))
        x0 = self.relu(self.conv1b(x0))
        x1 = self.pool(x0)
        x1 = self.relu(self.conv2a(x1))
        x1 = self.relu(self.conv2b(x1))
        x2 = self.pool(x1)
        x2 = self.relu(self.conv3a(x2))
        x2 = self.relu(self.conv3b(x2))
        x3 = self.pool(x2)
        x3 = self.relu(self.conv4a(x3))
        x3 = self.relu(self.conv4b(x3))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x3))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        self.h, self.w = h, w
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))
        # Keep the k keypoints with highest score
        if max_keypoints >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, max_keypoints)
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        scores = list(scores)

        # Compute the dense descriptors
        x3 = self.relu(self.convDa(x3))
        x3 = self.convDb(x3)
        x3 = torch.nn.functional.normalize(x3, p=2, dim=1)

        if mode == 'train':
            self.add_random_point(keypoints, scores)

        # Extract descriptors
        x3 = torch.stack([sample_descriptors_new(k[None], d[None], 8)[0] for k, d in zip(keypoints, x3)], 0)
        x2 = torch.stack([sample_descriptors_new(k[None], d[None], 4)[0] for k, d in zip(keypoints, x2)], 0)
        x1 = torch.stack([sample_descriptors_new(k[None], d[None], 2)[0] for k, d in zip(keypoints, x1)], 0)
        x0 = torch.stack([sample_descriptors_new(k[None], d[None], 1)[0] for k, d in zip(keypoints, x0)], 0)
        x0 = self.lin0(x0)
        x1 = self.lin1(x1)
        x2 = self.lin2(x2)
        descriptors = self.merge(torch.cat([x0, x1, x2, x3], 1))

        data.update({
            'keypoints0': torch.stack(keypoints, 0),
            'scores0': torch.stack(scores, 0),
            'descriptors0': descriptors,
        })

        x0 = self.relu(self.conv1a(data['image1']))
        x0 = self.relu(self.conv1b(x0))
        x1 = self.pool(x0)
        x1 = self.relu(self.conv2a(x1))
        x1 = self.relu(self.conv2b(x1))
        x2 = self.pool(x1)
        x2 = self.relu(self.conv3a(x2))
        x2 = self.relu(self.conv3b(x2))
        x3 = self.pool(x2)
        x3 = self.relu(self.conv4a(x3))
        x3 = self.relu(self.conv4b(x3))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x3))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if max_keypoints >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, max_keypoints)
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        scores = list(scores)
        # Compute the dense descriptors
        x3 = self.relu(self.convDa(x3))
        x3 = self.convDb(x3)
        x3 = torch.nn.functional.normalize(x3, p=2, dim=1)

        if mode == 'train':
            self.add_random_point(keypoints, scores)

        # Extract descriptors
        x3 = torch.stack([sample_descriptors_new(k[None], d[None], 8)[0] for k, d in zip(keypoints, x3)], 0)
        x2 = torch.stack([sample_descriptors_new(k[None], d[None], 4)[0] for k, d in zip(keypoints, x2)], 0)
        x1 = torch.stack([sample_descriptors_new(k[None], d[None], 2)[0] for k, d in zip(keypoints, x1)], 0)
        x0 = torch.stack([sample_descriptors_new(k[None], d[None], 1)[0] for k, d in zip(keypoints, x0)], 0)
        x0 = self.lin0(x0)
        x1 = self.lin1(x1)
        x2 = self.lin2(x2)
        descriptors = self.merge(torch.cat([x0, x1, x2, x3], 1))

        data.update({
            'keypoints1': torch.stack(keypoints, 0),
            'scores1': torch.stack(scores, 0),
            'descriptors1': descriptors,
        })
