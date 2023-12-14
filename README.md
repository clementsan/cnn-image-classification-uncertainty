# AI-based image classification tasks with uncertainty

Image classification tasks via CNN models, leveraging Monte Carlo dropout during inference to assess model uncertainty

Related paper: [Gal et al., Dropout as a bayesian estimation: representing model uncertainty in deep learning, ICML 2016](http://proceedings.mlr.press/v48/gal16.pdf)


## Analysis tasks

 - Image preprocessing
   - Generate small image tiles from larger acquisitions
   - Split dataset into training, validation and test sets, using stratification strategy at acquisition level
 - Image analysis
   - Neural network training
   - Neural network inference (direct)
   - Neural network inference with gradCAM visualization
   - Neural network inference with uncertainty assessment

## Notes

### Neural networks

Use of ResNet models

### Deep learning library

Use of fastai_v1 (2019)

