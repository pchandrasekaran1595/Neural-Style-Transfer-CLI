### **Neural Style Transfer Application**<br>

<br>

**[Model Reference](https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style)**

<br>

1. Install Python
2. Run `pip install virtualenv`
3. Run `make-env.bat` or `make-env-3.9.bat`

<br>

**Arguments**

<pre>
1. --mode | -m       - image or video or realtime
2. --model | -mo     - candy or mosaic or pointilism or rain-princess or udnie
3. --filename | -f   - Name of the image file (with extension)
4. --downscale | -ds - Downscale video by a factor before inference 
5. --negative | -n   - Do `Result = 255 - Result`
6. --save | -s       - Save the processed file (`filename - Result.png`)
</pre>