import torch
import torch.nn as nn
from loss import batch_episym
import time
class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class AFF(nn.Module):
    def __init__(self, channels, outchannel, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, outchannel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outchannel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, outchannel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outchannel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, xa):
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return wei

class sse(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(sse, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.conv0 = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, bias=True)
        self.in0 = nn.InstanceNorm2d(num_channels)
        self.bn0 = nn.BatchNorm2d(num_channels)
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        x = self.conv0(input_tensor)
        x = self.in0(x)
        x = self.bn0(x)
        input_tensor = self.relu(x)
        squeeze_tensor = input_tensor.view(
            batch_size, num_channels, -1).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        fs_out_1 = torch.mean(output_tensor, 1).unsqueeze(1)
        fs_out = self.sigmoid(fs_out_1)
        output_tensor = fs_out * output_tensor

        return output_tensor
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),  # b*c*n*1
            trans(1, 2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )

        # self.gcn = GCN_Block(points)
        # self.linear_0 = nn.Conv2d(points, 1, (1, 1))

        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x) #b*k*c*1
        out = out + self.conv2(out)  # + self.gcn(out, w0.detach())  #b*k*c*1
        out = self.conv3(out)   #b*c*k*1
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out  #b*c*k*1


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)#b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)# b*k*n
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3) #b*c*k*1
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)  # b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)  # b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class OABlock(nn.Module):
    def __init__(self, net_channels, depth=6, clusters=250):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(channels, l2_nums))
        self.up1 = diff_unpool(channels, l2_nums)
        self.l2 = nn.Sequential(*self.l2)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        self.aff = AFF(channels * 2, channels)
        self.pointcn = PointCN(channels * 2, channels)
        self.shot_cut = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, data):
        x1_1 = data
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = torch.cat([x1_1, x_up], dim=1)
        out1 = self.aff(out)
        out2 = self.pointcn(out)
        out = out1 * out2
        return out


class DSBlock(nn.Module):
    def __init__(self, net_channels, clusters=256, knn_num=6):
        nn.Module.__init__(self)
        channels = net_channels
        self.k_num = knn_num
        l2_nums = clusters
        self.down1 = diff_pool(channels, l2_nums)
        self.up1 = diff_unpool(channels, l2_nums)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        self.shot_cut = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.DGCNN_MAX_Block = DGCNN_MAX_Block(self.k_num, l2_nums)

    def forward(self, data):
        # data: b*c*n*1

        x1_1 = data
        x_down = self.down1(x1_1).transpose(1, 2)
        x2 = self.DGCNN_MAX_Block(x_down).transpose(1, 2)

        x_up = self.up1(x1_1, x2)
        out = torch.cat([x1_1, x_up], dim=1)
        return self.shot_cut(out)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # inner[32,2000,2000]???
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # xx[32,1,2000]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # distance[32,2000,2000]****?????

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k) [32,2000,9] [32,1000,6]

    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None):
    # x[32,128,2000,1],k=9
    # x[32,128,1000,1],k=6
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # x[32,128,2000]
    if idx is None:
        idx_out = knn(x, k=k)  # idx_out[32,2000,9]
    else:
        idx_out = idx

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx_out + idx_base  # idx[32,2000,9] ?32?????????

    idx = idx.view(-1)  # idx[32*2000*9] ?32???????? [32*1000*6]

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # x[32,2000,128]
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)  # feature[32,2000,9,128]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # x[32,2000,9,128]
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()  # feature[32,256,2000,9] ???
    return feature, idx_out  # 同时返回索引


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        # e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4 [32,1,500,4] logits[32,2,500,1]
    mask = logits[:, 0, :, 0]  # [32,500] logits????
    weights = logits[:, 1, :, 0]  # [32,500] logits????

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class DGCNN_MAX_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_MAX_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel
        self.edge_scorer = nn.Sequential(
            nn.Conv2d(in_channel * 2, 8, kernel_size=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )
        nn.init.constant_(self.edge_scorer[-1].bias, 3.0)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, (1, 1)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, (1, 1)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        self.reverse_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, (1, 1)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, (1, 1)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        self.count_emb_dim = 16
        self.count_embedder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, self.count_emb_dim, kernel_size=1)
        )

        self.conf_gate = nn.Sequential(
            nn.Conv2d(in_channel + self.count_emb_dim, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        if self.knn_num <= 9:
            optimal_bias = -1.5
        elif self.knn_num <= 12:
            optimal_bias = -2.2
        else:
            optimal_bias = -3.0

        nn.init.constant_(self.conf_gate[-1].bias, optimal_bias)

    def forward(self, features):
        B, C, N, _ = features.shape
        edge_feat, idx = get_graph_feature(features, k=self.knn_num)

        conv_out = self.conv(edge_feat)
        out_forward = conv_out.max(dim=-1, keepdim=False)[0]

        scores = torch.sigmoid(self.edge_scorer(edge_feat))

        idx_flat = idx.view(B, -1)
        center_feat = out_forward.unsqueeze(-1).expand(-1, -1, -1, self.knn_num)

        center_feat_weighted = center_feat * scores
        center_feat_flat = center_feat_weighted.reshape(B, C, -1)
        idx_expanded = idx_flat.unsqueeze(1).expand(-1, C, -1)

        out_reverse = torch.zeros(B, C, N, device=features.device, dtype=features.dtype)
        out_reverse = out_reverse.scatter_add(2, idx_expanded, center_feat_flat)

        scores_flat = scores.view(B, 1, -1)
        weighted_count = torch.zeros(B, 1, N, device=features.device, dtype=features.dtype)
        weighted_count = weighted_count.scatter_add(2, idx_flat.unsqueeze(1), scores_flat)

        clamp_count = weighted_count.clamp(min=1e-6)
        out_reverse = out_reverse / clamp_count.squeeze(1).unsqueeze(1)
        out_reverse = out_reverse.unsqueeze(-1)
        out_reverse = self.reverse_conv(out_reverse)  # [B, C, N, 1]

        count_log = torch.log(weighted_count.unsqueeze(-1) + 1.0)

        count_emb = self.count_embedder(count_log)  # [B, 16, N, 1]

        gate_input = torch.cat([out_reverse, count_emb], dim=1)

        confidence = torch.sigmoid(self.conf_gate(gate_input))
        out_reverse = out_reverse * confidence

        out_forward = out_forward.unsqueeze(-1)
        out = self.fusion(torch.cat([out_forward, out_reverse], dim=1))
        out = out + features

        return out



class GCN_Block(nn.Module):
    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)  # w[32,2000,1] ??0?1???
        A = torch.bmm(w.transpose(1, 2), w)  # A[32,1,1]
        return A  # TODO zfy

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()  # B=32,N=2000
        with torch.no_grad():
            A = self.attention(w)  # A[32,1,1]
            I = torch.eye(N).unsqueeze(0).to(x.device).detach()  # I[1,2000,2000]????
            A = A + I  # A[32,2000,2000]    TODO zfy
            D_out = torch.sum(A, dim=-1)  # D_out[32,2000]
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D)  # D[32,2000,2000]
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)  # L[32,2000,2000]
        out = x.squeeze(-1).transpose(1, 2).contiguous()  # out[32,2000,128]
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()  # out[32,128,2000,1]

        return out

    def forward(self, x, w):
        # x[32,128,2000,1],w[32,2000]
        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out


class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),  # 4?6 ? 128
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        self.sse = sse(self.out_channel)
        self.sse_g = sse(self.out_channel)
        self.gcn = GCN_Block(self.out_channel)

        self.embed_0 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            DGCNN_MAX_Block(int(self.k_num * 2), self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            OABlock(self.out_channel, clusters=256),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            OABlock(self.out_channel, clusters=128),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            DGCNN_MAX_Block(int(self.k_num * 2), self.out_channel),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.aff = AFF(out_channel, out_channel)
        self.pointcn = PointCN(out_channel, out_channel)
        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))
    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N, _ = x.size()
        indices = indices[:, :int(N * self.sr)]  # indices[32,1000]??????
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)  # y_out ????????[32,1000]
            w_out = torch.gather(weights, dim=-1, index=indices)  # w_out ??????w0[32,1000]
        indices = indices.view(B, 1, -1, 1)  # indices[32,1,1000,1]

        with torch.no_grad():
            x_out = torch.gather(x[:, :, :, :4], dim=2,
                                     index=indices.repeat(1, 1, 1, 4))  # x_out ??????x[32,1,500,4]
        feature_out = torch.gather(features, dim=2,
                                       index=indices.repeat(1, 128, 1, 1))  # feature_out ??????features[32,128,500,1]
        return x_out, y_out, w_out, feature_out

    def forward(self, x, y, sse):
        # x[32,1,2000,4],y[32,2000]
        # x_[32,1,1000,6],y1[32,1000]
        B, _, N, _ = x.size()
        out = x.transpose(1, 3).contiguous()  # contiguous??out?x??????out[32,4?6,2000,1]
        out = self.conv(out)  # out[32,128,2000,1]
        out = out + sse
        out = self.embed_0(out)  # NGCE+CGCE=ResNet * 3 + DGCNN_MAX + ResNet * 3 + OABlock + ResNet * 3
        w0 = self.linear_0(out).view(B, -1)  # w0[32,2000]

        out_g = self.gcn(out, w0.detach())  # GGCE
        out1 = out_g + out
        out2 = out
        out = self.aff(out1) * self.pointcn(out2) + out

        out = self.embed_1(out)  # CGCE+NGCE=ResNet * 2 + OABlock + ResNet * 2 + DGCNN_MAX + ResNet * 2
        w1 = self.linear_1(out).view(B, -1)  # w1[32,2000]

        if self.predict == False:  # ??????
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)  # w1??,w1_ds[32,2000],indices[32,2000]???
            w1_ds = w1_ds[:, :int(N * self.sr)]  # w1_ds[32,1000]??????? self.sr=0.5
            x_ds, y_ds, w0_ds, out= self.down_sampling(x, y, w0, indices, out)
            # x_ds[32,1,1000,4],y_ds[32,1000],w0_ds[32,1000],ds?????
            sse = self.sse(out)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds], sse
        else:  # ????????
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)  # w1??,w1_ds[32,1000],indices[32,1000]???
            w1_ds = w1_ds[:, :int(N * self.sr)]  # w1_ds[32,500]??????? self.sr=0.5
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out)
            # x_ds[32,1,500,4],y_ds[32,500],w0_ds[32,500],out[32,128,500,1]?????,ds?????
            out = self.embed_2(out)
            w2 = self.linear_2(out)  # [32,2,500,1]
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat
class DS_init(nn.Module):
    def __init__(self, out_channel, k, cluster):
        super(DS_init, self).__init__()
        self.in_channel = 4
        self.out_channel = out_channel
        self.k_num = k
        self.cluster = cluster
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),  # 4?6 ? 128
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )
        self.NGCE = DGCNN_MAX_Block(int(self.k_num * 2), self.out_channel)
        self.OA = OABlock(self.out_channel, clusters=self.cluster)
        self.sse = sse(self.out_channel)
    def forward(self, x):
        # x[32,1,2000,4],y[32,2000]
        B, _, N, _ = x.size()
        out = x.transpose(1, 3).contiguous()
        out = self.conv(out)
        out = self.NGCE(out)
        out = self.OA(out)
        sse1 = self.sse(out)
        return sse1

class CLNet(nn.Module):
    def __init__(self, config):
        super(CLNet, self).__init__()
        self.ds_init = DS_init(out_channel=128, k=4, cluster=64)
        self.ds_0 = DS_Block(initial=True, predict=False, out_channel=128, k_num=9,
                             sampling_rate=config.sr)  # sampling_rate=0.5
        self.ds_1 = DS_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr)

    def forward(self, x, y):
        # x[32,1,2000,4],y[32,2000]
        B, _, N, _ = x.shape
        sse1 = self.ds_init(x)
        x1, y1, ws0, w_ds0, sse2 = self.ds_0(x, y, sse1)  # ????x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)  # ??0?1???[32,1,1000,1]
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)  # ??0?1???[32,1,1000,1]
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)  # x_[32,1,1000,6] ??????????????

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1, sse2)  # x_[32,1,1000,6],y1[32,1000]
        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)  # y_hat??????
        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat

