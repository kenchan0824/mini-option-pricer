# imports
import sys
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QWidget, QMessageBox
from comp7405.basic import *
from comp7405.asian import *
from comp7405.basket import *
from comp7405.binomial import *

# load ui file
baseUIClass, baseUIWidget = uic.loadUiType('Qt/minipricer.ui')

# use loaded ui file in the logic class
class Logic(baseUIWidget, baseUIClass):

    def __init__(self, parent=None):
        super(Logic, self).__init__(parent)
        self.setupUi(self)
        self.btn_cal_price_1.clicked.connect(self.cal_basic_price)
        self.btn_cal_iv_1.clicked.connect(self.cal_basic_imp_vol)
        self.btn_cal_geom_2.clicked.connect(self.cal_asian_geom)
        self.btn_cal_arith_2.clicked.connect(self.cal_asian_arith)
        self.btn_cal_geom_3.clicked.connect(self.cal_basket_geom)
        self.btn_cal_arith_3.clicked.connect(self.cal_basket_arith)
        self.btn_cal_price_4.clicked.connect(self.cal_binomial_price)

    def exception_hook(self, exctype, value, traceback):
        # Prompt the error
        QMessageBox.warning(self, 'Warning', str(value))

    def cal_basic_price(self):
        S0 = self.dsb_S0_1.value()
        K = self.dsb_K_1.value()
        T = self.dsb_T_1.value()
        r = self.dsb_r_1.value() / 100.0
        q = self.dsb_q_1.value() / 100.0
        sigma = self.dsb_sigma_1.value() / 100.0
        C = black_scholes(S0, K, T, sigma, r, q, option_type='C')
        P = black_scholes(S0, K, T, sigma, r, q, option_type='P')
        self.lb_C_1.setText('%.3f' % C)
        self.lb_P_1.setText('%.3f' % P)

    def cal_basic_imp_vol(self):
        S0 = self.dsb_S0_1.value()
        K = self.dsb_K_1.value()
        T = self.dsb_T_1.value()
        r = self.dsb_r_1.value() / 100.0
        q = self.dsb_q_1.value() / 100.0
        if self.rb_call_1.isChecked():
            option_type = 'C'
        else:
            option_type = 'P'
        V = self.dsb_V_1.value()
        sigma = imp_vol(V, S0, K, T, r, q, option_type)
        self.lb_iv_1.setText('%.3f' % (sigma * 100.0))

    def cal_asian_geom(self):
        S0 = self.dsb_S0_2.value()
        n = self.sb_n_2.value()
        K = self.dsb_K_2.value()
        T = self.dsb_T_2.value()
        r = self.dsb_r_2.value() / 100.0
        sigma = self.dsb_sigma_2.value() / 100.0
        if self.rb_call_2.isChecked():
            option_type = 'C'
        else:
            option_type = 'P'
        m = self.sb_m_2.value()
        if self.rb_closed_2.isChecked():
            V = geom_asian_exact(S0, K, T, sigma, r, n, option_type)
            std = 0
        else:
            V, low, up = monte_carlo_asian(S0, K, T, sigma, r, n, option_type, sim_type='G', m=m)
            std = up - low
        self.lb_Vgeom_2.setText('%.3f' % V)
        self.lb_Vgeom_std_2.setText('%.3f' % std)

    def cal_asian_arith(self):
        S0 = self.dsb_S0_2.value()
        n = self.sb_n_2.value()
        K = self.dsb_K_2.value()
        T = self.dsb_T_2.value()
        r = self.dsb_r_2.value() / 100.0
        sigma = self.dsb_sigma_2.value() / 100.0
        if self.rb_call_2.isChecked():
            option_type = 'C'
        else:
            option_type = 'P'
        m = self.sb_m_2.value()
        if self.rb_yes_2.isChecked():
            sim_type = 'C'
        else:
            sim_type = 'A'
        V, low, up = monte_carlo_asian(S0, K, T, sigma, r, n, option_type, sim_type, m)
        std = up - low
        self.lb_Varith_2.setText('%.3f' % V)
        self.lb_Varith_std_2.setText('%.3f' % std)

    def cal_basket_geom(self):
        S1 = self.dsb_S1_3.value()
        S2 = self.dsb_S2_3.value()
        K = self.dsb_K_3.value()
        T = self.dsb_T_3.value()
        r = self.dsb_r_3.value() / 100.0
        sigma1 = self.dsb_sigma1_3.value() / 100.0
        sigma2 = self.dsb_sigma2_3.value() / 100.0
        rho = self.dsb_rho_3.value()
        if self.rb_call_3.isChecked():
            option_type = 'C'
        else:
            option_type = 'P'
        m = self.sb_m_3.value()
        if self.rb_closed_3.isChecked():
            V = geom_basket_exact(S1, S2, K, T, sigma1, sigma2, r, rho, option_type)
            std = 0
        else:
            V, low, up = monte_carlo_basket(S1, S2, K, T, sigma1, sigma2, r, rho, option_type,
                                            sim_type='G', m=m)
            std = up - low
        self.lb_Vgeom_3.setText('%.3f' % V)
        self.lb_Vgeom_std_3.setText('%.3f' % std)

    def cal_basket_arith(self):
        S1 = self.dsb_S1_3.value()
        S2 = self.dsb_S2_3.value()
        K = self.dsb_K_3.value()
        T = self.dsb_T_3.value()
        r = self.dsb_r_3.value() / 100.0
        sigma1 = self.dsb_sigma1_3.value() / 100.0
        sigma2 = self.dsb_sigma2_3.value() / 100.0
        rho = self.dsb_rho_3.value()
        if self.rb_call_3.isChecked():
            option_type = 'C'
        else:
            option_type = 'P'
        m = self.sb_m_3.value()
        if self.rb_yes_3.isChecked():
            sim_type = 'C'
        else:
            sim_type = 'A'
        V, low, up = monte_carlo_basket(S1, S2, K, T, sigma1, sigma2, r, rho, option_type,
                                        sim_type, m)
        std = up - low
        self.lb_Varith_3.setText('%.3f' % V)
        self.lb_Varith_std_3.setText('%.3f' % std)

    def cal_binomial_price(self):
        S0 = self.dsb_S0_4.value()
        K = self.dsb_K_4.value()
        T = self.dsb_T_4.value()
        r = self.dsb_r_4.value() / 100.0
        sigma = self.dsb_sigma_4.value() / 100.0
        if self.rb_call_4.isChecked():
            option_type = 'C'
        else:
            option_type = 'P'
        N = self.sb_N_4.value()
        V = binomial_tree(S0, K, T, sigma, r, option_type, N)
        self.lb_V_4.setText('%.3f' % V)


app = QtWidgets.QApplication(sys.argv)
#app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
ui = Logic(None)
ui.show()
# Set the exception hook to our wrapping function
sys.excepthook = ui.exception_hook
sys.exit(app.exec_())
