//TODO fix me when consts are available
var<private> rotations: array<u32, 8> = array<u32, 8>(13u, 15u, 26u, 6u, 17u, 29u, 16u, 24u);

@group(0)
@binding(0)
var<storage, read> key: array<u32, 2>;

@group(0)
@binding(1)
var<storage, read> data: array<u32>;

@group(0)
@binding(2)
var<storage, read_write> out: array<u32>;

fn rotate_left(v: u32, distance: u32) -> u32 {
    return (v << distance) | (v >> (32u - distance));
}

fn round(x: ptr<function, array<u32, 2>>, rotation: u32) {
    (*x)[0] += (*x)[1];
    (*x)[1] = rotate_left((*x)[1], rotation);
    (*x)[1] ^= (*x)[0];
}

@compute
@workgroup_size(256)
fn threefry(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let half = arrayLength(&data) / 2u;
    if global_id.x < half {
        var x: array<u32, 2>;
        var ks: array<u32, 3>;

        ks[2] = 0x1BD11BDAu;

        ks[0] = key[0];
        x[0] = data[idx];
        ks[2] = ks[2] ^ key[0];

        ks[1] = key[1];
        x[1] = data[idx + half];
        ks[2] = ks[2] ^ key[1];

        x[0] = x[0] + ks[0];
        x[1] = x[1] + ks[1];

        for (var i = 0u; i < 4u; i++) {
            round(&x, rotations[i]);
        }

        x[0] = x[0] + ks[1];
        x[1] = x[1] + ks[2] + 1u;
        for (var i = 4u; i < 8u; i++) {
            round(&x, rotations[i]);
        }

        x[0] = x[0] + ks[2];
        x[1] = x[1] + ks[0] + 2u;
        for (var i = 0u; i < 4u; i++) {
            round(&x, rotations[i]);
        }

        x[0] = x[0] + ks[0];
        x[1] = x[1] + ks[1] + 3u;
        for (var i = 4u; i < 8u; i++) {
            round(&x, rotations[i]);
        }

        x[0] = x[0] + ks[1];
        x[1] = x[1] + ks[2] + 4u;
        for (var i = 0u; i < 4u; i++) {
            round(&x, rotations[i]);
        }

        out[idx] = x[0] + ks[2];
        out[idx + half] = x[1] + ks[0] + 5u;
    }
}