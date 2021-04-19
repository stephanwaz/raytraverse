Directional Sampling Overview
=============================

.. raw:: html

    <iframe width="800" height="450" src="https://www.youtube.com/embed/slXjIQKGK3k?start=17782" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

(starting at 4:56:25)

Transcript
----------

**1. Title Slide**

Hello, my name is Stephen Wasilewski and I am presenting some work I have
prepared along with my co-authors. Raytraverse is a new method that guides the
sampling process of a daylight simulation.

**2. The Daylight Simulation Process**

To understand how this method can enhance the daylight simulation process, it
is useful to view the process by parts.

**2.b**

The model describes how geometry, materials, and light sources are represented.

**2.c**

Sampling determines how the analysis dimensions are subdivided into discrete
points to simulate.

**2.d**

These views rays are solved for by a renderer, yielding a radiance or an
irradiance value for each view ray.

**2.e**

This output is evaluated according to some metric or otherwise preparing the
data for interpretation.

**3. Assumptions**

To make a viable workflow, each of these parts require (whether explicitly or
implicitly) a number of assumptions that define the limitations and
opportunities of the method. To explain this in practical terms, here are three
examples of well known climate based modeling methods for visual comfort.

**4. CBDM Methods for Visual Comfort: Ev based**

Illuminance based methods, including DGPs (simplified Daylight Glare
Probability), limit the directional sampling resolution to a single sample per
view direction in order to efficiently sample a larger number of positions and
sky conditions throughout a space.

Unfortunately: Even if the employed rendering method perfectly captures the
true Illuminance, as a model for discomfort glare it fails to account for
scenes where the dominant driver of discomfort is contrast based or due to
small bright sources in an otherwise dim scene.

**5. CBDM Methods for Visual Comfort: 3/5 Phase**

The 3-phase and 5-phase methods focus on the model and render steps. These
methods fix the implementations of the material and sky models by discretizing
the transmitting materials and sky dome in order to replace some steps of the
rendering process with a matrix multiplication.

**6. CBDM Methods for Visual Comfort: eDGPs**

Like the 5-phase method, The enhanced-simplified daylight glare probability
method, developed to overcome the limitations of illuminance only metrics, uses
separate sampling and rendering assumptions for the indirect contribution and
direct view rays. The adaptation level is captured by an illuminance value, but
glare sources are identified with an image calculated for direct view ray
contributions only.

**7. Existing Options For Sampling a Point**

In all of these methods, the sampling is treated as a fixed assumption.

**7.b**

Either directional sampling is directly integrated into an illuminance by the
renderer,

**7.c**

or a high resolution image is generated.

**7.d**

This is because at intermediate image resolutions the accuracy of the results
can be worse than an illuminance sample, and are unreliable for capturing
contrast effects due to small sources.

**7.e**

So unlike sampling positions or timesteps which can be set at arbitrary spacing
and easily tuned to the needs of the analysis, directional sampling is much
more of an all or nothing choice; where the additional insights offered by an
image can require 1 million times more data than a point sample. But is this
really necessary?

**7.f**

Whether through direct image interpretation or any of the commonly used glare
metrics, the critical information embedded in an HDR image is usually
simplified to a small set of sources and background, each with a size,
direction and intensity. We cannot directly sample this small set of rays
because we do not know these important directions ahead of time, but how close
can we get?

**7.g**

The raytraverse method provides a means to bridge the gap between point samples
and high resolution images, allowing for a tunable tradeoff between simulation
time and accuracy.

Our approach is structured by a wavelet space representation of the directional
sampling. It works by applying a set of filters to an image to locate these
important details.

**8. Wavelet Decomposition**

To match our sampling space, we apply these filters to a square image space
based on the Shirley-Chiu disk to square transform, which preserves adjacency
and area, both necessary for locating true details.

**8.b**

For each level of the decomposition, The high pass filters, applied across each
axis (vertical, horizontal, and in combination) isolate the detail in the
image, and the low pass filter performs an averaging yielding an image of half
the size. This process is repeated, applying the high pass filters to the
approximation, down to some base resolution. Each level of the decomposition
stores the relative change in intensity at a particular resolution (or
frequency).

**8.c**

The total size of the output arrays is the same as the original, and can be
used to perfectly recover the original signal through the inverse transform.

The benefit to compression comes from the fact that the magnitude of the detail
coefficients effectively rank the data in terms of their contribution to the
reconstruction. By thresholding the coefficients, less important data can be
discarded.

**8.d**

Even after discarding over 99% of the wavelet coefficients, the main image
details are recoverable and only some minor artifacts have been introduced.

This property, that the wavelet coefficients rank the importance of samples at
given resolutions, makes detail coefficients useful for guiding the sampling of
view rays from a point.

**9. Reconstruction Through Sampling**

This process works as follows:

Beginning with a low resolution initial sampling the large scale features of
the scene are captured.

Mimicking the wavelet transform, We apply a set of filters to this estimate and
then use the resulting detail coefficients both to find an appropriate number
of samples, and as probability distribution for the direction of these samples.

The new sample results returned by the renderer are used to update the
estimate, which is lifted to a higher resolution.

This process is repeated up to a maximum resolution, equivalent to (or higher
than) what a full resolution image might be rendered at.

**10. Component Sampling**

There are some cases where the wavelet based sampling will not find important
details, such as specular views and reflections of the direct sun. Fortunately,
because our method uses sky-patch coefficients to efficiently capture arbitrary
sky conditions (similar to 3 phase and others), we can structure the simulation
process in such a way to compensate for these misses. I refer you to our paper
for details on how this works.

**11. Results**

Instead, I’ll spend my remaining time sharing a few examples of scenes captured
with: our approach, a high resolution reference and a matching uniform
resolution image to demonstrate the benefits of variable sampling.

In addition to image reconstructions, the relative deviation from the reference
is shown for vertical illuminance (characterizing energy conservation) and UGR
(Unified Glare Rating, characterizing contrast), relative errors greater than
10% are highlighted in red.

This very glary scene highlights the different paths that light takes from the
sun to the eye, including direct views, rough specular and diffuse reflections
of the sun and sky. While the deviation in the low resolution image is unlikely
to change a prediction in this case, the large errors show a failure case for
uniform low-res sampling.

**11.b**

A more complex, but also more likely scenario is that roller shades will be
closed. While there are open questions on how to evaluate the specular
transmission of such materials, raytraverse does not introduce any substantial
new errors to this process.

**11.c**

Raytraverse performs similarly well for partially open venetian blinds.

**11.d**
Including deeper in a space where the floor reflection dominates.

**11.e**

Raytraverse, without virtual sources or other rendering tricks, handles the
case of specular reflections of the direct sun, a difficult problem for low
resolution sampling.

**11.f**

One case that we would expect raytraverse to struggle with would be a high
frequency pattern like the dot frit shown here. And while the sampling does
miss parts of the pattern, especially the lower contrast areas, enough of the
detail is caught to meaningfully understand the image and, because of the
direct sun view sampling, maintains high accuracy.

**11.g**

In cases where more image fidelity is desired, raytraverse can be tuned to
increase the sampling rate with a proportional increase in simulation time, but
in our paper we show that the low sampling rates previously shown achieve a
high level of accuracy for field of view metrics.

**12. Thank you**

Thank you for watching my presentation.
