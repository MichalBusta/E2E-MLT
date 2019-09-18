
## Train Data Format

code support 2 kind of annotations:
 - YOLO like format (https://github.com/MichalBusta/E2E-MLT/blob/3110671b178a868bc33c44442983170d57d6de89/data_gen.py#L39):
 
 ```
 cls, x, y, w, h, angle, GT_TEXT
 ```
  where:
    - cls is not used
    - [x, y] is center of bounding box (in relative coordinates: x = X / img.cols y = Y / img.rows )
    - w, h is bounding box width and height normalized by image diagonal (w = W / sqrt( img.cols * img.cols + img.rows * img.rows))
    - angle of rotated bounding box in radians
 
 - ICDAR like format (https://github.com/MichalBusta/E2E-MLT/blob/3110671b178a868bc33c44442983170d57d6de89/data_gen.py#L86):

``` 
 x1,y1,x2,y2,x3,y3,x4,y4,GT_TEXT
```
 where [x1,y1] is coordinate (in pixels) of bottom left corner of text bounding box. The bounding box is annoted in clock wise order (bottom left, top left, top right, top bottom).
 
 naming convetion is defined here: https://github.com/MichalBusta/E2E-MLT/blob/3110671b178a868bc33c44442983170d57d6de89/data_gen.py#L616 
