// Alogorithm obtained from https://github.com/golang/go/blob/go1.21.6/src/math/erfinv.go

@group(0)
@binding(0)
var <storage, read> input_values: array<f32>;

@group(0)
@binding(1)
var <storage, read_write> output_values: array<f32>;

const a0 = 1.1975323115670912564578e0;
const a1 = 4.7072688112383978012285e1;
const a2 = 6.9706266534389598238465e2;
const a3 = 4.8548868893843886794648e3;
const a4 = 1.6235862515167575384252e4;
const a5 = 2.3782041382114385731252e4;
const a6 = 1.1819493347062294404278e4;
const a7 = 8.8709406962545514830200e2;
const b0 = 1.0000000000000000000e0;
const b1 = 4.2313330701600911252e1;
const b2 = 6.8718700749205790830e2;
const b3 = 5.3941960214247511077e3;
const b4 = 2.1213794301586595867e4;
const b5 = 3.9307895800092710610e4;
const b6 = 2.8729085735721942674e4;
const b7 = 5.2264952788528545610e3;
// Coefficients for approximation to erf in 0.85 < |x| <= 1-2*exp(-25);
const c0 = 1.42343711074968357734e0;
const c1 = 4.63033784615654529590e0;
const c2 = 5.76949722146069140550e0;
const c3 = 3.64784832476320460504e0;
const c4 = 1.27045825245236838258e0;
const c5 = 2.41780725177450611770e-1;
const c6 = 2.27238449892691845833e-2;
const c7 = 7.74545014278341407640e-4;
const d0 = 1.4142135623730950488016887e0;
const d1 = 2.9036514445419946173133295e0;
const d2 = 2.3707661626024532365971225e0;
const d3 = 9.7547832001787427186894837e-1;
const d4 = 2.0945065210512749128288442e-1;
const d5 = 2.1494160384252876777097297e-2;
const d6 = 7.7441459065157709165577218e-4;
const d7 = 1.4859850019840355905497876e-9;
// Coefficients for approximation to erf in 1-2*exp(-25) < |x| < 1;
const e0 = 6.65790464350110377720e0;
const e1 = 5.46378491116411436990e0;
const e2 = 1.78482653991729133580e0;
const e3 = 2.96560571828504891230e-1;
const e4 = 2.65321895265761230930e-2;
const e5 = 1.24266094738807843860e-3;
const e6 = 2.71155556874348757815e-5;
const e7 = 2.01033439929228813265e-7;
const f0 = 1.414213562373095048801689e0;
const f1 = 8.482908416595164588112026e-1;
const f2 = 1.936480946950659106176712e-1;
const f3 = 2.103693768272068968719679e-2;
const f4 = 1.112800997078859844711555e-3;
const f5 = 2.611088405080593625138020e-5;
const f6 = 2.010321207683943062279931e-7;
const f7 = 2.891024605872965461538222e-15;
const Ln2 = 0.693147180559945309417232121458176568;


// Erfinv returns the inverse error function of x.
//
// Special cases are:
//
//	Erfinv(1) = +Inf
//	Erfinv(-1) = -Inf
//	Erfinv(x) = NaN if x < -1 or x > 1
//	Erfinv(NaN) = NaN
@compute
@workgroup_size(256)
fn erfinv(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x < arrayLength(&output_values) {
        var x = input_values[global_id.x];

        if x != x || x <= -1 || x >= 1 {
            if x == 1 {
                output_values[global_id.x] = bitcast<f32>(2139095040u);
            } else if x == -1 {
                output_values[global_id.x] = bitcast<f32>(4286578688u);
            } else {
                output_values[global_id.x] = bitcast<f32>(2143289344u);
            }
        } else {
            var sign = false;
            if x < 0 {
                x = -x;
                sign = true;
            }

            if x <= 0.85 { // |x| <= 0.85
                var r = 0.180625 - 0.25 * x * x;
                var z1 = ((((((a7 * r + a6) * r + a5) * r + a4) * r + a3) * r + a2) * r + a1) * r + a0;
                var z2 = ((((((b7 * r + b6) * r + b5) * r + b4) * r + b3) * r + b2) * r + b1) * r + b0;
                output_values[global_id.x] = (x * z1) / z2;
	        } else {
                var z1: f32;
                var z2: f32;
                var r = sqrt(Ln2 - log(1.0 - x));
                if r <= 5.0 {
                    r -= 1.6;
                    z1 = ((((((c7 * r + c6) * r + c5) * r + c4) * r + c3) * r + c2) * r + c1) * r + c0;
                    z2 = ((((((d7 * r + d6) * r + d5) * r + d4) * r + d3) * r + d2) * r + d1) * r + d0;
                } else {
                    r -= 5.0;
                    z1 = ((((((e7 * r + e6) * r + e5) * r + e4) * r + e3) * r + e2) * r + e1) * r + e0;
                    z2 = ((((((f7 * r + f6) * r + f5) * r + f4) * r + f3) * r + f2) * r + f1) * r + f0;
                }
                output_values[global_id.x] = z1 / z2;
            }

            if sign {
                output_values[global_id.x] = -output_values[global_id.x];
            }
        }
    }
}