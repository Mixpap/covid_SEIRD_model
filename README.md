# covid_SEIRD_model
A Full SEIRD model the covid spread in Greece. 
The model Predicts:
1. Real Cases
2. Deaths
3. Official/Real cases

The model parameters are estimated comparing with the official announced cases, official announced deaths and other data, as the detection of the virus in the sewers. Some of the parameters are time-dependent (R0(t), Test Bias Factor A(t) and mortallity m(t)) using the pchip interpolation (Piecewise Cubic Hermite Interpolating Polynomial).
