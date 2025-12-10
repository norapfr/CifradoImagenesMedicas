from core import *

def encode():
    im, rounds, l = get_data()
    original_shape = im.shape
    np.save("shape.npy", np.array(original_shape))


    params = generate_params()
    X0 = generate_X0()
    save_keys(params, X0)

    A = {
        "A_omega": cat_map_4d_matrix(*params["p_omega"]),
        "A_psi": cat_map_4d_matrix(*params["p_psi"]),
        "A_a2_primary": cat_map_2d_matrix(*params["p_a2_primary"]),
        "A_a2_secondary": cat_map_2d_matrix(*params["p_a2_secondary"])
    }

    Cblocks = encrypt_image(im, rounds, l, A, X0)
    save_image(Cblocks, l, original_shape, "encrypted.png")
