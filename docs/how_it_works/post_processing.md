# Post processing
Here we will explain how the post processing works in Biom3d, from the steps to transform the models output to a masks and other filters.

## Post processing function :
The post processing function works in several steps :
1. **Return logit :** If a specific parameter is set to true, it will directly return the result of the model without converting it to a mask. This is not the default behaviour. 
2. **Translation to mask :** The output is translated into a usable mask. For that there are 3 options :
   1. The default is to use softmax. It is a mathematical function that will transform the different values of a voxel across the channels in probability and keep the highest. It is the default behaviour and is great when doing multiclass.
   2. Another possibility is sigmoid. It is another mathematical function that will polarize the values of the voxel across the channel and then a thresold is applied (thresold value being 0.5). This function should be used for images with only one class (+ background).
   3. The last possibility is to force softmax, in case where the model has been trained with sigmoid. We compute both sigmoid and softmax as shown above and we keep the softmax part where the sigmoid is at 1. 
3. We have a mask and now there are 2 possibles filters to remove noise :
   1. Keep big only : This filter compute the volume of the different segmented object and do a otsu thresolding to keep only the largest of them.
   2. Keep biggest only : This filter keep only the biggest **centered** object, it is used in case where you know that you have 1 segmented object per image and that it is centered.
   If both filter are selected, you will have a warning and it will use Keep biggest only.

The image is always resample to it's original shape before pre-preprocessing, if you decide to use the option `--skip preprocessing`, you must assure that the image has `original_shape` in its metadata and is in `(C,D,H,W)` for 3D aornd `(C,1,H,W)` for 2D.

## Modifying the paramters :
There are two way to modify the post procesing behaviour : by calling the function with specific parameter, that require to develop your own python script, or to modify the `config.yaml` file, if you want to do that before the training, you will have to . 

The default look of the post processing part in the config file is :
```yaml
POSTPROCESSOR:
  fct: Seg
  kwargs:
    use_softmax: true
    keep_biggest_only: false
    keep_big_only: false
```
or 
```python
POSTPROCESSOR = Dict(
    fct="Seg",
    kwargs=Dict(
        use_softmax=USE_SOFTMAX,
        keep_biggest_only=False,
        keep_big_only=False,
    ),
)
```
Now let's see how to change it. When you modify the yaml, ensure that you follow the indentation :
```yaml
POSTPROCESSOR:
  fct: Seg
  kwargs:
    use_softmax: true
    variable: value <- will work
variable: value <- will crash
    keep_biggest_only: false
    keep_big_only: false
```
Also, there is no space before the `:`, but always one after. 
When you modify the python config, you need to to follow python syntaxe. 

If you have any doubt about how to write a value or on syntaxe, check how we do in our [documentation](../tuto/config.md), the default configs or on Python or yaml documentation.

- The `fct` is a postprocessing function defined by [`register.py`](../api/register.py). For the moment there is only one prediction function so there is no use to modify it.
- If you want to skip postprocessing and get the raw output, you will need to add a new parameter :
    ```yaml
    POSTPROCESSOR:
    fct: Seg
    kwargs:
        use_softmax: true
        return_logit: true
        keep_biggest_only: false
        keep_big_only: false
    ```
    or 
    ```python
    POSTPROCESSOR = Dict(
        fct="Seg",
        kwargs=Dict(
            use_softmax=USE_SOFTMAX,
            return_logit=True, # Don't forget the comma
            keep_biggest_only=False,
            keep_big_only=False,
        ),
    )
    ```
- To use sigmoid instead of softmax, set `use_softmax: false` (in yaml) or `USE_SOFTMAX=False` (in python, it should be around line 114)
- If you trained on sigmoid and want to force softmax, do the same thing as above and add the parameter `force_softmax:true` :
  ```yaml
    POSTPROCESSOR:
    fct: Seg
    kwargs:
        use_softmax: false
        force_softmax: true
        keep_biggest_only: false
        keep_big_only: false
    ```
    or 
    ```python
    USE_SOFTMAX=False
    POSTPROCESSOR = Dict(
        fct="Seg",
        kwargs=Dict(
            use_softmax=USE_SOFTMAX,
            force_softmax=True, # Don't forget the comma
            keep_biggest_only=False,
            keep_big_only=False,
        ),
    )
    ```
- If you want to use `keep_big_only` or `keep_biggest_only`, you don't necessarly need to modify the config files as it is an option in our GUI, however this option hasn't been integrated in our CLI yet. If you want to modify the config here how to :
  ```yaml
    POSTPROCESSOR:
    fct: Seg
    kwargs:
        use_softmax: true
        keep_biggest_only: false # Replace False by True
        keep_big_only: false # Replace False by True
    ```
    or 
    ```python
    POSTPROCESSOR = Dict(
        fct="Seg",
        kwargs=Dict(
            use_softmax=USE_SOFTMAX,
            keep_biggest_only=False, # Replace False by True
            keep_big_only=False, # Replace False by True
        ),
    )
    ```