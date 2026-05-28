# CUTTING To do list

### First pages:
1. Look at TA after
2. Make contents and acronym page 2 columned?
3. Less focus on power in intro. Almost assume SNNs have less power, but how do we make SNNs actually good and useable? Is using biology a viable option or are SNNs just better by themselves.
4. Less focus on edge devices too. Focus on the SNN
    a) I.e. SNNs have been proposed as a solution, how do we make these good
    b) Need a good reference for this

### background:
5. Massively cut down auditory cues for background. Remove useless figures. 
    a) One section for all auditory cues. State how they work
    b) Description box or bullet points
6. You show the CF-FM signals in the background so cut later
7. Merge auditory cues for bats into auditory cues
8. Remove elementary theory of biological neurons
9. Remove vocalisation pathway
10. Have a functional modelling approach to the biology. Link to model immediately. Quick bit about how it works, then simply “this was modelled as”. Will be easier to understand. (Then will need some time series examples for these, maybe some diagrams)
11. Focus on what is modelled, not the “in reality this happens” unless reader would have thought of it.
12. State the basics if needed, but only explain and elaborate on the more complex
13. Shrink down for now and build up later if time. Just state the basics, eg surrogate gradients, show concepts and diagrams as everyone hates maths.
14. Put the note on hardware in current models, as this shows why spikes and rates can be used.
15. Replace current models with actual research

### Modelling
16. Use the modelling methodology as your disclaimer that more work was done, with only key results and final refined models shown.
17. Need to state parameters of testing
    a) For key parameters, such as constrained space, state in main body eg inline
    b) For experimental setup of testing, state at start of results section
    c) Put exhaustive list of parameters in the appendix as a list of tables grouped by their area/function. The experiments should be repeatable to a reader.
18. Either shrink down the neuron testing and coincidence detector section, or use this to help show the fundamental building blocks with diagrams so that the assessor can understand the rest. At the moment it is in no mans land with reasonably redundant results. Maybe explain clearly the coincidence detector in the background so you can state it as a one liner everywhere else.
19. Maybe replace initial work stuff with a walkthrough example. Effectively primes the reader for what they are about to see and so they know how the pieces fit together before seeing how the pieces were made.
20. Maybe get rid of note on optimisation. I personally think it is a lovely little detail that would be incredibly useful if someone wanted to recreate this, but is actually useless for results of this project.
21. Move noise to the actual environment pre HRTF
22. Show noise signal example of reflected signal to build some intuition of what we are analysing.
23. Shrink the half wave rectification and LIF encoding massively. People know what HWR is and LIFs will be explained in the background. Whole part can be explained by a one liner
24. Mathematical equivalence is completely supplementary. Replace with a one liner saying this can be built using neurons, potentially showing the equivalence immediately and in one line then continue with IIR
25. Dynamic thresholding can be shrunk a lot. It isnt that complicated. Maybe include the second plot but get rif of first for sure.
26. Use paragraphs to label parts that you are talking about. Then state its function with an equation. Remove some of the prose as it is just fluffy at the moment
27. Never text wrap, but potentially make minipages with a direct longer description of a figure if needed
28. Get rid of full LIF explanation. LIFs will be explained in background.
29. Maybe explain the mexican hat once, and then just state it afterwards. I.e. “this is reused later in the report and will be referred to as Mexican Hat sharpening”
30. Remove true working of LNTB MSO line. Only confuses the assessor. They need to understand your model and your model only after 1 read. If this was an actual paper maybe but this is for a grade!
31. Maybe make a “readouts” section. This shifts how the CANN is framed and directly compares it to the COM. Also means you dont have to repat the COM 3 times. Just state at the end of each thing that you get an activity population readout that is fed into the SC to decode. Also reduces focus on CANN, meaning you can potentially remove it from background as it is unimportant, just state it is a particular arrangements of excitatory and inhibitory neurons that can smooth and stabilise neural population codes. This also makes the story more logical, as the CANN is simply used as a readout for the network.
32. Selected ear for elevation can be a one liner
33. Good detail on elevation, but is far too long. This is not a developmental log book. Change to final model and what it has, not how it was developed
34. Remove DCN inspired refinements. NO MENTION. No one liner. They are not in the model. Yes this is cool maths and work that is down the drain, deal with it
35. Redo whole CANN section. “Readout mechanism”. Immediately frame it as a readout only and as the input for the fusion network.
36. Dont be afraid to mention results early!!! It is not a story it is a report.
37. Shrink explanation to just stating the most basic form of the CANN, stating it had some corrections to account for edge conditions.
38. Output and coordinate decoding is pretty much the whole section
39. Get rid of Gain section… this is sad as I was proud of this. It just is so irrelevant to the main model and is only useful for the CANN. Dont confuse the reader. 
40. Just replace all the other stuff about CANNs with simply it is a way of having a model with steady state stability but transient growth.
41. Need more details on architecture etc for training models. Leave this section reasonably long as it should not be seen as an afterthought, but what the whole thing has been building up to. Stress this fact. 
42. Rework trainable section to be more consistent. State common features, then differences. Very split at the moment and the link is less obvious

### Results
43. State testing setup at start of results so it is clear what is being tested. I.e. the models were run for 2800 samples in these conditions yadadada
44. Then have a brief mention of the metrics used but you dont need to explain them as these are well known
45. Need error bounds too
46. State why the constrained area is what it is. State that testing wider angles would be fair but the final models were tuned to the cone, and that this is what bats have as well as it is directional. State also that the distance minimum was based kind of on the pulse. State that down to 0 was tested but due to no inhibition being simulated at the offset this is a redundant result. Also that going further than 5m was tested and there was a clear trend of worse results at a distance, but this result is redundant as it is directly dependent on the SNR of the signal, which has been set based on ideas from nature, but these limits will need to be found from hardware implementations, and so range constraints are arbitrary
47. Remove diag vs reflected CANN and simplify to just CANN-based. Note that you could still used your optimised version, and just state the simplest form of the CANN model in the modelling section for simplicity
48. Maybe show the distance pathway results in the worked example instead, and just use the table as the results. The graphs are literally just an example here anyway.
49. Test if the COM readout is actually over time or whether it is end of simulation like you say it is. If it is actually over time, rethink whether the CANN is needed at all, and can just be stated as a way of balancing the COM readout with lateral inhibition.
50. Keep warping in results section as I like the slight bit of story telling here, i.e. the reader seeing how predictable the results are and maybe coming up with the idea themself, then seeing that I did the same.
51. Need better plots that are more condensed and show error better, potentially a comparative bar chart, or an average error across the coordinate range. The latter could be very nice actually. Maybe average across distance bins too for azimuth since there will be trends and may show how the warping is dependent on distance.
52. Remove developmental results for elevation. Remove diagonal vs reflected CANN results too. Maybe just use diagonal for simplicity.
53. General note on CANN results, maybe just use diagonal model with [I,0] input, simple recurrence and no FI optimisation. Results dont seem to differ much anyway and readout across time seems to actually be more stable
54. Just use full noise results for CANN? The drop off scatter plot is quite nice acc. Need to decide which I want, and whether to compare each pathway one by one or all 3 at once.
55. I like the split of trainable baselines and then using the best one against the feature based training. 
56. Need much more analysis and discussion. Very much stating the facts here and not giving insight.
57. I like the braking up of conclusion, just refine these to make them tighter
58. Future work is 2.5 pages right now. Cut down almost all of this into a general ideas and concise, well grounded suggestions of implementation. DO not make it a list of ideas, Put it before the conclusion potentially?
