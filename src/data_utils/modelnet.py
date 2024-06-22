# Modified from the codebase for Structuring Representations Using Group Invariants:
# https://proceedings.neurips.cc/paper_files/paper/2022/hash/dcd297696d0bb304ba426b3c5a679c37-Abstract-Conference.html
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('agg')


def read_off(filename):
    with open(filename, 'r') as file:
        # if 'OFF' != file.readline().strip():
        #     raise('Not a valid OFF header')
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


def render(vertices, triangles, R, lim=0.95):
    vertices = (R @ vertices.T).T
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    fig = plt.figure(figsize=(4, 4), dpi=128)
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.axis('off')
    ax.plot_trisurf(
        x, z, triangles, y, shade=True, color=(0.5, 0.5, 0.5),
        edgecolor='none', linewidth=0, antialiased=False, alpha=1.0
    )

    images = []

    ax.view_init(elev=0, azim=0)
    # taken from https://stackoverflow.com/a/61443397/3090085
    with io.BytesIO() as io_buf:
        fig.savefig(io_buf, facecolor='black', format='raw', dpi=128)
        io_buf.seek(0)
        images.append(
            np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                       newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        )
    # -------------------------------------------------------

    ax.view_init(elev=0, azim=90)
    # taken from https://stackoverflow.com/a/61443397/3090085
    with io.BytesIO() as io_buf:
        fig.savefig(io_buf, facecolor='black', format='raw', dpi=128)
        io_buf.seek(0)
        images.append(
            np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                       newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        )
    # -------------------------------------------------------

    ax.view_init(elev=90, azim=0)
    # taken from https://stackoverflow.com/a/61443397/3090085
    with io.BytesIO() as io_buf:
        fig.savefig(io_buf, facecolor='black', format='raw', dpi=128)
        io_buf.seek(0)
        images.append(
            np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                       newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        )
    # -------------------------------------------------------

    plt.close('all')
    images = [np.array(Image.fromarray(x).resize((48, 48), resample=Image.BICUBIC).convert('L')) for x in images]
    return np.stack(images, axis=0)
