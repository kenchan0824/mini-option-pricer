Mini Options Pricer
===

A desktop application built with PyQt. Demonstrated to price American options, Asian options, and Basket options with Binomial method and Monte-Carlo simulation. Also finding implied volatility of European options with Newton-Raphson method.

Let's challenge how less codes you can write than mine to code up the Binomial method. My implementation is highly vectorized. I have borrowed the ideas from the deep learning frameworks to process the lattice layers forward and then backward. It has also proofed to be the most robust and fastest execution in the class demo.

List of Files
===
|Folder|File Name|Description|
|---|---|---|
|`<root>`|`main.py`|The GUI of the application, written in PyQt5.
||`Test_Basic.ipynb`|The testing Jupyter Notebook for basic.py
||`Test_Asian.ipynb`|The testing Jupyter Notebook for asian.py
||`Test_Basket.ipynb`|The testing Jupyter Notebook for basket.py
||`Test_Binomial.ipynb`|The testing Jupyter Notebook for binomial.py
|`<comp7405>`|`basic.py`|European option pricing and implied volatility calculation.
||`asian.py`|Monte Carlo method for Asian option pricing.
||`basket.py`|Monte Carlo method for Basket option pricing.
||`binomial.py`|Binomial method for American option pricing.
|`<Qt>`|`minipricer.ui`|The user interface definition required by main.py

Instructions
===
- Install the following packages
	- python (3.6.2)
	- pyqt (5.6.0)
	- numpy (1.12.1)
	- scipy (0.19.0)

- Type `python main.py` to start

