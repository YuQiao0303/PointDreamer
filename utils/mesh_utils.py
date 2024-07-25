
import numpy as np
import torch
from trimesh import grouping
from trimesh.geometry import faces_to_edges

def subdivide_with_uv( vertices, faces, face_uv_idx,uvs, face_index=None):
    """
    Modified from:
    https://github.com/mikedh/trimesh/blob/85b4bd1f410d8d8361009c6f27266719a3d2b97d/trimesh/remesh.py#L15

    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    face_uv_idx : (F, 3) int
      Indexes of uvs for each vertex of each face
    uvs : (uv_num, 2) float
      UV coordinates
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces
   


    Returns
    ----------
    new_vertices : (q, 3) float
      Vertices in space
    new_faces : (p, 3) int
      Remeshed faces
    new_uvs : (uv_num, 2) float
      Remeshed uvs
    new_face_uv_idx : (F, 3) int
      Indexes of uvs for each vertex of each face
    """
    if face_index is None:
        face_mask = np.ones(len(faces), dtype=bool)
    else:
        face_mask = np.zeros(len(faces), dtype=bool)
        face_mask[face_index] = True

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_mask]
    face_uv_subset = face_uv_idx[face_mask]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    # get new uv
    edges_uv = np.sort(faces_to_edges(face_uv_subset), axis=1)
    unique_uv, inverse_uv = grouping.unique_rows(edges_uv)
    mid_uv = uvs[edges_uv[unique_uv]].mean(axis=1)
    mid_idx_uv = inverse_uv.reshape((-1, 3)) + len(uvs)

    # the new faces_subset with correct winding
    f = np.column_stack(
        [
            faces_subset[:, 0],
            mid_idx[:, 0],
            mid_idx[:, 2],
            mid_idx[:, 0],
            faces_subset[:, 1],
            mid_idx[:, 1],
            mid_idx[:, 2],
            mid_idx[:, 1],
            faces_subset[:, 2],
            mid_idx[:, 0],
            mid_idx[:, 1],
            mid_idx[:, 2],
        ]
    ).reshape((-1, 3))

    f_uv = np.column_stack(
    [
        face_uv_subset[:, 0],
        mid_idx_uv[:, 0],
        mid_idx_uv[:, 2],
        mid_idx_uv[:, 0],
        face_uv_subset[:, 1],
        mid_idx_uv[:, 1],
        mid_idx_uv[:, 2],
        mid_idx_uv[:, 1],
        face_uv_subset[:, 2],
        mid_idx_uv[:, 0],
        mid_idx_uv[:, 1],
        mid_idx_uv[:, 2],
    ]
    ).reshape((-1, 3))

    # add the 3 new faces_subset per old face all on the end
    # by putting all the new faces after all the old faces
    # it makes it easier to understand the indexes
    new_faces = np.vstack((faces[~face_mask], f))

    new_face_uv_idx = np.vstack((face_uv_idx[~face_mask], f_uv))
    # stack the new midpoint vertices on the end
    new_vertices = np.vstack((vertices, mid))

    new_uvs = np.vstack((uvs, mid_uv))

    return new_vertices,new_faces,new_uvs,new_face_uv_idx

def test_subdivide_with_uv(input_file):
    def savemeshtes2(pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, fname):
      import os
      fol, na = os.path.split(fname)
      na, _ = os.path.splitext(na)

      matname = os.path.join(fol, f'model_normalized.mtl')  #matname = '%s/%s.mtl' % (fol, na)
      fid = open(matname, 'w')

      fid.write('newmtl material_0\n')
      fid.write('Kd 1 1 1\n')
      fid.write('Ka 0 0 0\n')
      fid.write('Ks 0.4 0.4 0.4\n')
      fid.write('Ns 10\n')
      fid.write('illum 2\n')
      fid.write('map_Kd %s.png\n' % na)
      fid.close()
      print('save',matname)
      ####

      fid = open(fname, 'w')
      fid.write('mtllib %s.mtl\n' % na)

      for pidx, p in enumerate(pointnp_px3):
          pp = p
          fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))

      for pidx, p in enumerate(tcoords_px2):
          pp = p
          fid.write('vt %f %f\n' % (pp[0], pp[1]))

      fid.write('usemtl material_0\n')
      for i, f in enumerate(facenp_fx3):
          f1 = f + 1
          f2 = facetex_fx3[i] + 1
          fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
      fid.close()

    import kaolin as kal
    import torch
    import kiui
    device = 'cpu'
    mesh = kal.io.obj.import_mesh(input_file, with_materials=True)


    vertices = mesh.vertices.to(device)
    faces = mesh.faces.to(device)
    uvs = mesh.uvs.to(device)
    face_uvs_idx = mesh.face_uvs_idx.to(device)
    materials = [m['map_Kd'].permute(2, 0, 1).unsqueeze(0).to(device).float() / 255. if 'map_Kd' in m else
                      m['Kd'].reshape(1, 3, 1, 1).to(device)
                      for m in mesh.materials]
    atlas_img = materials[0]
    kiui.lo(atlas_img)
    kiui.lo(vertices)
    kiui.lo(faces)
    kiui.lo(uvs)
    kiui.lo(face_uvs_idx)
    face_vert_uvs = uvs[face_uvs_idx]
    kiui.lo(face_vert_uvs)
    new_vertices,new_faces,new_uvs,new_face_uv_idx = subdivide_with_uv(vertices, faces, face_uvs_idx,uvs)

    kiui.lo(new_vertices)
    kiui.lo(new_faces)
    kiui.lo(new_uvs)
    kiui.lo(new_face_uv_idx)

    savemeshtes2(
        new_vertices, # pointnp_px3
        new_uvs, # tcoords_px2
        new_faces, # facenp_fx3
        new_face_uv_idx, # facetex_fx3
        'temp.obj') # fname

if __name__ == '__main__':
  test_subdivide_with_uv('temp/meshes/model.obj')
    