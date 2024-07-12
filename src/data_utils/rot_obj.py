from modelnet import read_off, render
from tqdm import tqdm
import numpy as np
import argparse
from scipy.stats import special_ortho_group
from scipy.linalg import expm, logm


def generate_data_SymReg(data_path, num_samples, num_actions):
    vertices, triangles = read_off(data_path)
    vertices -= np.mean(vertices, axis=0)
    vertices /= np.mean(np.linalg.norm(vertices, axis=1))
    action_list = [expm(0.2 * logm(special_ortho_group.rvs(3))) for _ in range(num_actions)]
    init_list = [special_ortho_group.rvs(3) for _ in range(num_samples)]
    images = np.empty((num_actions, num_samples, 3, 48, 48), dtype=np.uint8)
    for i in tqdm(range(num_samples)):
        for j in range(num_actions):
            R = init_list[i] @ action_list[j]
            images[j, i] = render(vertices, triangles, R)
    np.save(data_path + '/images', images)
    print("Generated Images")


def generate_data_LieGAN(data_path, num_samples, num_actions, axis, output_path, output_name):
    vertices, triangles = read_off(data_path)
    vertices -= np.mean(vertices, axis=0)
    vertices /= np.mean(np.linalg.norm(vertices, axis=1))
    if num_actions > 1:
        action_list = [expm(0.2 * logm(special_ortho_group.rvs(3))) for _ in range(num_actions)]
    if axis is None:
        init_list = [special_ortho_group.rvs(3) for _ in range(num_samples)]
        so3rep_list = []
        for R in init_list:
            g = logm(R)
            so3rep_list.append(np.array([g[2, 1], g[0, 2], g[1, 0]]))
    else:
        init_list = []
        so3rep_list = []
        if axis == 'random':
            R_offset = special_ortho_group.rvs(3)
            direction = np.random.rand(3)
            direction /= np.linalg.norm(direction)
            print(f'Random offset: {R_offset}')
            print(f'Random axis direction: {direction}')
        for i in range(num_samples):
            R = special_ortho_group.rvs(3)
            g = logm(R)
            if axis == 'x':
                g[1, 0] = g[0, 1] = g[0, 2] = g[2, 0] = 0
            elif axis == 'y':
                g[2, 1] = g[1, 2] = g[1, 0] = g[0, 1] = 0
            elif axis == 'z':
                g[2, 1] = g[1, 2] = g[0, 2] = g[2, 0] = 0
            elif axis == 'random':
                # random axis direction
                g = np.zeros((3, 3))
                w = np.random.rand() * 2 * np.pi
                g[0, 1], g[1, 0] = w * direction[2], -w * direction[2]
                g[2, 0], g[0, 2] = w * direction[1], -w * direction[1]
                g[1, 2], g[2, 1] = w * direction[0], -w * direction[0]
            R = expm(g)
            if axis == 'random':
                R = R @ R_offset
            init_list.append(R)
            if axis == 'random':
                so3rep_list.append(np.array([w]))
            else:
                so3rep_list.append(np.array([g[2, 1], g[0, 2], g[1, 0]]))
    if num_actions > 1:
        init_list_1, so3rep_list_1 = [[] for _ in init_list], [[] for _ in init_list]
        for i, R in enumerate(init_list):
            for A in action_list:
                init_list_1[i].append(A @ R)
                g = logm(A @ R)
                so3rep_list_1[i].append(np.array([g[2, 1], g[0, 2], g[1, 0]]))
        np.save(f'{output_path}/{output_name}_rot.npy', so3rep_list_1)
        images = np.empty((num_samples, num_actions, 3, 48, 48), dtype=np.uint8)
        for i in tqdm(range(num_samples)):
            for j in range(num_actions):
                images[i, j] = render(vertices, triangles, init_list_1[i][j])
        np.save(f'{output_path}/{output_name}.npy', images)
    else:
        np.save(f'{output_path}/{output_name}_rot.npy', so3rep_list)
        images = np.empty((num_samples, 3, 48, 48), dtype=np.uint8)
        for i in tqdm(range(num_samples)):
            images[i] = render(vertices, triangles, init_list[i])
        np.save(f'{output_path}/{output_name}.npy', images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='shelf')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_actions', type=int, default=1)
    parser.add_argument('--axis', type=str, default=None)
    parser.add_argument('--name', type=str, default='train')
    args = parser.parse_args()
    output_path = '../data/rotobj'
    if args.dataset == 'chair':
        data_path = '../data/rotobj/chair.off'
        generate_data_LieGAN(data_path, args.num_samples, args.num_actions, args.axis, output_path, args.name)
    elif args.dataset == 'shelf':
        data_path = '../data/rotobj/bookshelf.off'
        generate_data_LieGAN(data_path, args.num_samples, args.num_actions, args.axis, output_path, args.name)
    else:
        raise NotImplementedError
