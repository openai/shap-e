# Model Card: Shap-E

This is the official codebase for running the latent diffusion models described in [Shap-E: Generating Conditional 3D Implicit Functions](https://arxiv.org/abs/2305.02463). These models were trained and released by OpenAI. Following [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993), we're providing some information about how the models were trained and evaluated.

# Model Details

Shap-E includes two kinds of models: an encoder and a latent diffusion model.

 1. **The encoder** converts 3D assets into the parameters of small neural networks which represent the 3D shape and texture as an implicit function. The resulting implicit function can be rendered from arbitrary viewpoints or imported into downstream applications as a mesh.
 2. **The latent diffusion model** generates novel implicit functions conditioned on either images or text descriptions. As above, these samples can be rendered or exported as a mesh. Specifically, these models produce latents which must be linearly projected to get the final implicit function parameters. The final projection layer of the encoder is used for this purpose.

Like [Point-E](https://github.com/openai/point-e/blob/main/model-card.md), Shap-E can often generate coherent 3D objects when conditioned on a rendering from a single viewpoint. When conditioned on text prompts directly, Shap-E is also often capable of producing recognizable objects, although it sometimes struggles to combine multiple objects or concepts.

Samples from Shap-E are typically lower fidelity than professional 3D assets and often have rough edges, holes, or blurry surface textures.

# Model Date

April 2023

# Model Versions

The following model checkpoints are available in this repository:

 * `transmitter` - the encoder and corresponding projection layers for converting encoder outputs into implicit neural representations.
 * `decoder` - just the final projection layer component of `transmitter`. This is a smaller checkpoint than `transmitter` since it does not include parameters for encoding 3D assets. This is the minimum required model to convert diffusion outputs into implicit neural representations.
 * `text300M` - the text-conditional latent diffusion model.
 * `image300M` - the image-conditional latent diffusion model.

# Paper & Samples

[Paper link](https://arxiv.org/abs/2305.02463) / [Samples](samples.md)

# Training data

The encoder and image-conditional diffusion models are trained on the [same dataset as Point-E](https://github.com/openai/point-e/blob/main/model-card.md#training-data). However, a few changes to the post-processing were made:

 * We rendered 60 views (instead of 20) of each model when computing point clouds, to avoid small cracks.
 * We produced 16K points in each point cloud instead of 4K.
 * We simplified the lighting and material setup to only include diffuse materials.

For our text-conditional diffusion model, we expanded our dataset with roughly a million more 3D assets. Additionally, we collected 120K captions from human annotators for a high-quality subset of our 3D assets.

# Evaluated Use

We release these models with the intention of furthering progress in the field of generative modeling. However, we acknowledge that our models have certain constraints and biases, which is why we advise against employing them for commercial purposes at this time. We are aware that the utilization of our models could extend to areas beyond our expectations, and defining specific criteria for what is considered suitable for "research" purposes presents a challenge. Specifically, we advise caution when using these models in contexts that demand high accuracy, where minor imperfections in the generated 3D assets could have adverse consequences.

Specifically, these models have been evaluated on the following tasks for research purposes:

 * Generating 3D renderings or meshes conditioned on single, synthetic images
 * Generating 3D renderings or meshes conditioned on text descriptions

# Performance & Limitations

Our image-conditional model has only been evaluated on a highly specific distribution of synthetic renderings. Even in these cases, the model still sometimes fails to infer the correct occluded parts of an object or produces geometry that is inconsistent with the given rendered images. These failure modes are similar to those of Point-E. The resulting 3D assets often have rough edges, holes, or blurry surface textures.

Our text-conditional model can also produce a somewhat large and diverse vocabulary of objects. This model is often capable of producing objects with requested colors and textures, and sometimes even combining multiple objects. However, it often fails for more complex prompts that require placing multiple objects in a scene or binding attributes to objects. It also typically fails to produce a desired number of objects when a certain quantity is requested.

We find that our text-conditional model can sometimes produce samples which reflect gender biases. For example, samples for "a nurse" typically have a different body shape than samples for "a doctor". When probing for potential misuses, we also found that our text-conditional model is capable of producing 3D assets related to violence, such as guns or tanks. However, the resulting quality of these samples is poor enough that they look unrealistic and toy like.

As with Point-E, our dataset consists of many simple, cartoonish 3D assets, and our generative models are prone to imitating this style.

We believe our models will have many potential use cases. For example, our text-conditional model could enable users to quickly produce many 3D assets, allowing for rapid prototyping for computer graphics applications or 3D printing.

The use of 3D printing in concert with our models could potentially be harmful, for example if used to create dangerous objects or fabricate tools or parts that are deployed without external validation.

Generative 3D models share many challenges and constraints with image generation models. This includes the tendency to generate content that may be biased or detrimental, as well as the potential for dual-use applications. As the capabilities of these models evolve, further investigation is required to gain a clearer understanding of how these risks manifest.
