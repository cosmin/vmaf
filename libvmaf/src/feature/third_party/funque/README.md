# `funque`

`<vmaf_dir>` used in explaining is the root directory of this workspace. 

## Prerequisites

This is same as that of libvmaf, it can be found in the file `<vmaf_dir>/libvmaf/README.md`. Refer to [`libvmaf`](../../../../README.md) Prerequisites section.

## Compile

From `<vmaf_dir>`, run:

```
meson setup libvmaf/build libvmaf --buildtype release -Denable_integer_funque=true
```

(add `--Denable_float_funque=true` flag if you want to use the floating-point feature extractors for funque.)

Build with (from <vmaf_dir>):

```
ninja -vC build
```

## FUNQUE Models

If no model file is passed, then the default `libvmaf` model will be used. 

The `funque` models are present in `<vmaf_dir>/model`
- `funque_integer.json` to extract integer features
- `funque_float.json` to extract floating-point features

### Configurations in FUNQUE Model file

Most of the configurations are similar to that of `libvmaf`. 

The extra configurations used in `funque` are resizing the input frames & different VIF levels.
- Resizer: Can be configured using `model_dict["feature_opts_dicts"][0]["enable_resize"]` parameter. This has to be either true or false
- VIF levels: Can be configured using `model_dict["feature_opts_dicts"][0]["vif_levels"]` parameter. Minimum expected value is 2 & maximum expected value is 4

## Usage

The usage is similar to that of `libvmaf`. Refer command-line tool [`vmaf`](../../../../tools/README.md) Usage section.
Bitdepths 8 and 10 are supported for `funque` feature extractors.

## Example

The following example shows how to get `funque` features for a pair of yuv inputs ([`src01_hrc00_576x324.yuv`](https://github.com/Netflix/vmaf_resource/blob/master/python/test/resource/yuv/src01_hrc00_576x324.yuv), [`src01_hrc01_576x324.yuv`](https://github.com/Netflix/vmaf_resource/blob/master/python/test/resource/yuv/src01_hrc01_576x324.yuv))

Run the following command from `<vmaf_dir>`
```shell script
./libvmaf/build/tools/vmaf \
    --reference src01_hrc00_576x324.yuv \
    --distorted src01_hrc01_576x324.yuv \
    --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
    --model path=model/funque_integer.json \
    --output output.csv --csv
```

The output feature values will be stored in the `output.csv` file in csv format. All the `libvmaf` output formats are supported.