from manim import *
import math 

#changes according to scene before running in terminal 

config.background_color = WHITE #BLACK #"#1E1E1E" #WHITE
#config.background_color = BLACK #BLACK #"#1E1E1E" #WHITE
#config.background_color = "#1E1E1E" #BLACK #"#1E1E1E" #WHITE

class nneq(Scene):
    def construct(self):
        eq = MathTex(r"f(x) = wx + b", font_size=200, color=BLACK)
        eq1 = MathTex(r"f(x) = w_{j,k} x_{k, i} + b_j", font_size=150, color=BLACK)

        self.play(Write(eq, run_time=3.5))
        self.wait()
        self.play(TransformMatchingShapes(eq,eq1))
        self.wait()

class Function(Scene):
    def construct(self):
        ax = Axes(               
            x_range=[-10,10],
            y_range=[-7,7],
            tips=False,
            axis_config={
                'color': BLACK,
            }
        )
        graph1 = ax.plot(lambda x: x, x_range=[-10,10], color=MAROON_C)
        graph2 = ax.plot(lambda x: x*x, x_range=[-10,10], color=MAROON_C)
        graph3 = ax.plot(lambda x: math.sin(x), x_range=[-10,10], color=MAROON_C)

        label1 = MathTex(r"y = f(x) = x", font_size=50, color=MAROON_C, substrings_to_isolate="x").to_edge(UL)
        label2 = MathTex(r"y = f(x) = x^2", font_size=50,color=MAROON_C, substrings_to_isolate="x").to_edge(UL)
        label3 = MathTex(r"y = f(x) = sin(x)", font_size=50,color=MAROON_C, substrings_to_isolate="x").to_edge(UL)

        label1.set_color_by_tex('x', TEAL_C)
        label2.set_color_by_tex('x', TEAL_C)
        label3.set_color_by_tex('x', TEAL_C)


        np.random.seed(1)
        X = np.arange(-10,10,0.01)
        X = np.random.choice(X,50)
        Y = np.array([math.sin(x) for x in X])
        dot_axes = ax.plot_line_graph(x_values=X,y_values=Y, vertex_dot_style={'color':TEAL_C}, vertex_dot_radius=0.05)

        self.play(Create(ax))
        self.play(AnimationGroup(Write(graph1, run_time=2), Write(label1, run_time=1)))
        self.play(AnimationGroup(Transform(graph1, graph2, run_time=2), Transform(label1, label2, run_time=1)))
        self.play(AnimationGroup(Transform(graph1, graph3, run_time=2), Transform(label1, label3, run_time=1)))
        self.wait(1)
        self.play(AnimationGroup(FadeOut(graph1, run_time=1), FadeOut(label1, run_time=1)))
        self.wait(1)
        self.play(FadeIn(dot_axes['vertex_dots']), run_time=2)
        self.play(Create(graph3), run_time=5)
        self.wait(5)


class Neuron(Scene):
    def construct(self):
        ax = Axes(               
            x_range=[-10,10],
            y_range=[-7,7],
            tips=False,
            axis_config={
                'color': BLACK,
            }
        )
        W = ValueTracker(1)
        b = ValueTracker(0)

        W_num = DecimalNumber().set_color(MAROON_E).to_edge(UL).shift(DOWN*0.8).shift(RIGHT*1.8)
        b_num = DecimalNumber().set_color(MAROON_E).to_edge(UL).shift(DOWN*1.6).shift(RIGHT*1.8)

        W_num.add_updater(lambda m: m.set_value(W.get_value()))
        b_num.add_updater(lambda m: m.set_value(b.get_value()))

        pre_label = MathTex(r"f(x) &= Wx + b", font_size=200, color=MAROON_C, substrings_to_isolate="x")
        pre_label.set_color_by_tex('x', TEAL_C)
        pre_label.save_state()


        label = MathTex(r"f(x) &= Wx + b \\ W &= \\ b &=", font_size=50, color=MAROON_C, substrings_to_isolate="x").to_edge(UL)
        label.set_color_by_tex('x', TEAL_C)

        
        graph = FunctionGraph(lambda x: W.get_value() * x + b.get_value(), color=MAROON_C)
        graph.add_updater(lambda func: func.become(FunctionGraph(lambda x: W.get_value() * x + b.get_value(), color=MAROON_C)))

        self.play(Write(pre_label, run_time=2) )
        self.play(Transform(pre_label, label, run_time=1) )
        self.play(AnimationGroup(Create(ax), Write(W_num), Write(b_num), Write(graph)))
        self.wait()
        self.play(W.animate.set_value(5), run_time=1)
        self.play(W.animate.set_value(-5), run_time=2)
        self.play(W.animate.set_value(1), run_time=1)
        self.play(b.animate.set_value(3), run_time=1)
        self.play(b.animate.set_value(-3), run_time=2)
        self.play(b.animate.set_value(0), run_time=2)
        self.wait()
        self.play(AnimationGroup(FadeOut(W_num),FadeOut(b_num), FadeOut(graph), FadeOut(ax)))
        self.play(Restore(pre_label))
        self.wait()


class Test(Scene):
    def construct(self):
        ax = Axes(               
            x_range=[-3,2],
            y_range=[-10,4],
            x_length=5,
            y_length=6,
            tips=False,
            axis_config={
                'color': BLACK,
            }
        )
        ax.to_edge(RIGHT)
        graph = ax.plot(lambda x: - x**3 - 2 * x**2 + 2 * x + 1, x_range=[-4,4], color=BLACK)
        graph2 = ax.plot(lambda x:  -17.7052 * x - 5.5595 , x_range=[-4,4], color=TEAL_C)
        

        Z1 = MathTex(r"Z_1 = W_1x + b_1", font_size=50, color=MAROON_C)
        Z2 = MathTex(r"Z_2 = W_2x + b_2", font_size=50, color=MAROON_C)
        Z3 = MathTex(r"Z_3 = W_3x + b_3", font_size=50, color=MAROON_C)
        Z4 = MathTex(r"Z_4 = W_4x + b_4", font_size=50, color=MAROON_C)

        Z1_n = MathTex(r"Z_1 = W_1x + b_1", font_size=50, color=MAROON_C)
        Z2_n = MathTex(r"Z_2 = W_2x + b_2", font_size=50, color=MAROON_C)
        Z3_n = MathTex(r"Z_3 = W_3x + b_3", font_size=50, color=MAROON_C)
        Z4_n = MathTex(r"Z_4 = W_4x + b_4", font_size=50, color=MAROON_C)

        Z1_num = MathTex(r"Z_1 = 2.82x -2.75", font_size=50, color=MAROON_C)
        Z2_num = MathTex(r"Z_2 = -1.64x + 0.18", font_size=50, color=MAROON_C)
        Z3_num = MathTex(r"Z_3 = 0.26x - 0.59", font_size=50, color=MAROON_C)
        Z4_num = MathTex(r"Z_4 = -2.09x - 4.05", font_size=50, color=MAROON_C)
    

        Z = VGroup(Z1, Z2, Z3, Z4).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)
        Z_n = VGroup(Z1_n, Z2_n, Z3_n, Z4_n).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)
        Z_num = VGroup(Z1_num, Z2_num, Z3_num, Z4_num).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)
        Z_num = VGroup(Z1_num, Z2_num, Z3_num, Z4_num).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)

        Z_sum =  MathTex(r"Z = Z_1 + Z_2 + Z_3 + Z_4", font_size=50, color=TEAL_C).next_to(Z_num,DOWN)
        Z_sumed = MathTex(r"Z = -17.7 x - 5.55", font_size=80, color=TEAL_C).to_edge(LEFT)
    

        self.play(Create(ax))
        self.play(Create(graph),)
        self.play(Write(Z, run_time= 3),)
        self.wait()
        #self.play(Transform(Z, Z_n))
        self.wait()
        self.play(Transform(Z, Z_num))
        self.wait()
        self.play(Write(Z_sum, run_time=1))
        self.wait()
        self.play(AnimationGroup(Transform(Z_sum, Z_sumed), FadeOut(Z) , Create(graph2, run_time=3)))
        self.wait()


class Activation(Scene):
    def construct(self):
        
        # Sigmoid
        ax1 = Axes(x_range=[-4,4], y_range=[-2,2], x_length=4, y_length=3, tips=False, stroke_width=1, axis_config={'color': WHITE})
        ax1.to_edge(RIGHT)
        rect1 = Rectangle(height=3*1.1, width=4*1.1, color=WHITE, fill_opacity=0, stroke_width=0.5).move_to(ax1)
        graph1 = ax1.plot(lambda x: 1 / (1 + math.exp(-x)), color=MAROON_C)
        label1 = MathTex(r"f(x) = sigmoid(x)", color=MAROON_C,  font_size=32).move_to(rect1, DOWN).shift(DOWN)

        # ReLU
        ax2 = Axes(x_range=[-4,4], y_range=[-2,2], x_length=4, y_length=3, tips=False, stroke_width=1, axis_config={'color': WHITE})
        rect2 = Rectangle(height=3*1.1, width=4*1.1, color=WHITE, fill_opacity=0, stroke_width=0.5).move_to(ax2)
        graph2 = ax2.plot(lambda x: max(0,x), color=TEAL_C, x_range=[-2,2])
        label2 = MathTex(r"f(x) = ReLU(x)", color=TEAL_C,  font_size=32).move_to(rect2, DOWN).shift(DOWN)

        # tanh
        ax3 = Axes(x_range=[-4,4], y_range=[-2,2], x_length=4, y_length=3, tips=False, stroke_width=1, axis_config={'color': WHITE})
        ax3.to_edge(LEFT)
        rect3 = Rectangle(height=3*1.1, width=4*1.1, color=WHITE, fill_opacity=0, stroke_width=0.5).move_to(ax3)
        graph3 = ax3.plot(lambda x: math.tanh(x), color=MAROON_C)
        label3 = MathTex(r"f(x) = tanh(x)", color=MAROON_C,  font_size=32).move_to(rect3, DOWN).shift(DOWN)

        # main ReLU
        ax = Axes(x_range=[-4,4], y_range=[-2,2], x_length=4*2, y_length=3*2, tips=False, stroke_width=1, axis_config={'color': WHITE})


        self.play(AnimationGroup(Create(rect1), Create(rect2), Create(rect3)))
        self.play(AnimationGroup(Create(ax1), Create(ax2), Create(ax3)))
        self.play(AnimationGroup(Write(label1, run_time=1), Create(graph1, run_time=1)))
        self.play(AnimationGroup(Write(label2, run_time=1), Create(graph2, run_time=1)))
        self.play(AnimationGroup(Write(label3, run_time=1), Create(graph3, run_time=1)))
        self.wait()
        self.play(FadeOut(ax1,ax3,graph1,graph3,label1,label3,rect1,rect3))
        self.play(FadeOut(ax2,graph2,label2,rect2))
        self.wait()


class ReLU(Scene):
    def construct(self):
        ax = Axes(x_range=[-6,6], y_range=[-4,4], x_length=8, y_length=6, tips=False, stroke_width=1, axis_config={'color': WHITE}).move_to(RIGHT*2)


        relu0 = Tex(r"f(x) = Rectified Linear Unit", font_size=100, color=MAROON_C,substrings_to_isolate="x")
        relu1 =  MathTex(r"f(x) &= ReLU(x)", font_size=65, color=MAROON_C,substrings_to_isolate="x").to_corner(UL)
        relu2 =  MathTex(r"f({{x}}) &= max({{x}},0)", font_size=65, color=MAROON_C).to_corner(UL)
        relu3 =  MathTex(r"f({{x}}) &= max(W{{x}} + b,0) \\ W &=  \\ b &=", font_size=65, color=MAROON_C).to_corner(UL)


        relu0.set_color_by_tex('x', TEAL_C)
        relu1.set_color_by_tex('x', TEAL_C)
        relu2[1].set_color(TEAL_C)
        relu2[3].set_color(TEAL_C)
        relu3[1].set_color(TEAL_C)
        relu3[3].set_color(TEAL_C)



        W = ValueTracker(1)
        b = ValueTracker(0)

        W_num = DecimalNumber(font_size=65).set_color(TEAL_C).move_to(relu3, DOWN).shift(UP).shift(LEFT*0.6)  # .to_edge(UL).shift(DOWN*0.8).shift(RIGHT*1.8)
        b_num = DecimalNumber(font_size=65).set_color(TEAL_C).move_to(relu3, DOWN).shift(LEFT*0.6)  #.to_edge(UL).shift(DOWN*1.6).shift(RIGHT*1.8)

        W_num.add_updater(lambda m: m.set_value(W.get_value()))
        b_num.add_updater(lambda m: m.set_value(b.get_value()))


        graph = ax.plot(lambda x: max(W.get_value() * x + b.get_value(),0), color=TEAL_C)
        graph.add_updater(lambda func: func.become(ax.plot(lambda x: max(W.get_value() * x + b.get_value(),0), color=TEAL_C)))


        self.play(Write(relu0))
        self.play(AnimationGroup(Transform(relu0,relu1), Create(ax)))

        self.wait()
        self.play(Transform(relu0,relu2))
        self.wait()
        self.play(Transform(relu0,relu3))
        self.play(AnimationGroup(Create(graph), Write(W_num), Write(b_num)))
        self.wait()
        self.play(W.animate.set_value(5), run_time=1)
        self.play(W.animate.set_value(0), run_time=2)
        self.play(W.animate.set_value(1), run_time=1)
        self.play(b.animate.set_value(3), run_time=1)
        self.play(b.animate.set_value(-3), run_time=2)
        self.play(b.animate.set_value(0), run_time=2)
        self.wait()


class model(Scene):
    def construct(self):
        ax = Axes(               
            x_range=[-3,2],
            y_range=[-10,4],
            x_length=5,
            y_length=6,
            tips=False,
            axis_config={
                'color': WHITE,
            }
        )
        ax.to_edge(RIGHT)

        graph = ax.plot(lambda x: - x**3 - 2 * x**2 + 2 * x + 1, x_range=[-4,4], color=WHITE)
        
        graph2_1 = ax.plot(lambda x: 4.34 * max(-2.09 * x - 4.05, 0) ,
        x_range=[-4,4], color=TEAL_C)
        graph2_2 = ax.plot(lambda x: - 1.52 * max(-1.64 * x + 0.18,0) + 4.34 * max(-2.09 * x - 4.05, 0),
        x_range=[-4,4], color=TEAL_C)
        graph2_3 = ax.plot(lambda x: -3.85 * max(2.82*x -2.75,0)- 1.52 * max(-1.64 * x + 0.18,0) + 4.34 * max(-2.09 * x - 4.05, 0),
        x_range=[-4,4], color=TEAL_C)
        graph2_4 = ax.plot(lambda x: -3.85 * max(2.82*x -2.75,0)- 1.52 * max(-1.64 * x + 0.18,0) - 1.04 * max(0.26 * x  - 0.59,0) + 4.34 * max(-2.09 * x - 4.05, 0) ,
        x_range=[-4,4], color=TEAL_C)
        
        graph2 = ax.plot(lambda x: -3.85 * max(2.82*x -2.75,0)- 1.52 * max(-1.64 * x + 0.18,0) - 1.04 * max(0.26 * x  - 0.59,0) + 4.34 * max(-2.09 * x - 4.05, 0) +1.09,
        x_range=[-4,4], color=TEAL_C)


        graph3 = ax.plot(lambda x: 0.58 * max(-0.1*x -0.33,0) + 3.12 * max(-1.71 * x -2.85,0) + 1.33 * max(-0.61 * x + 1.229,0) - 1.27 * max(-1.15 * x + 0.12, 0)
         + 0.01 * max(-0.13*x -0.6 ,0) - 1.53 * max(-1.42 * x + 0.24,0) + 3.8 * max(-1.54 * x - 3.65,0) - 3.83 * max(2.7 * x -2.85, 0)   -0.09,
        x_range=[-4,4], color=TEAL_C)

        Z1 = MathTex(r"Z_1 &= (W_1x + b_1)", font_size=50, color=MAROON_C)
        Z2 = MathTex(r"Z_2 &= (W_2x + b_2)", font_size=50, color=MAROON_C)
        Z3 = MathTex(r"Z_3 &= (W_3x + b_3)", font_size=50, color=MAROON_C)
        Z4 = MathTex(r"Z_4 &= (W_4x + b_4)", font_size=50, color=MAROON_C)
        
        A1 = MathTex(r"A_1 &= ReLU(Z_1)", font_size=50, color=MAROON_C)
        A2 = MathTex(r"A_2 &= ReLU(Z_2)", font_size=50, color=MAROON_C)
        A3 = MathTex(r"A_3 &= ReLU(Z_3)", font_size=50, color=MAROON_C)
        A4 = MathTex(r"A_4 &= ReLU(Z_4)", font_size=50, color=MAROON_C)

        A1_e = MathTex(r"A_1 &= ReLU(W_1x + b_1)", font_size=50, color=MAROON_C)
        A2_e = MathTex(r"A_2 &= ReLU(W_2x + b_2)", font_size=50, color=MAROON_C)
        A3_e = MathTex(r"A_3 &= ReLU(W_3x + b_3)", font_size=50, color=MAROON_C)
        A4_e = MathTex(r"A_4 &=  ReLU(W_4x + b_4)", font_size=50, color=MAROON_C)

        A1_num = MathTex(r"A_1 &= ReLU(2.82x -2.75)", font_size=50, color=MAROON_C)
        A2_num = MathTex(r"A_2 &= ReLU(-1.64x + 0.18)", font_size=50, color=MAROON_C)
        A3_num = MathTex(r"A_3 &= ReLU(0.26x  - 0.59)", font_size=50, color=MAROON_C)
        A4_num = MathTex(r"A_4 &= ReLU(-2.09x - 4.05)", font_size=50, color=MAROON_C)

        eq = MathTex(r"A &= ReLU(Z) \\ &= ReLU( {{\text{Weights} }} * x + {{\text{biases} }} )", font_size=60, color=MAROON_C).to_edge(LEFT) #, substrings_to_isolate=['Weights','bias'])
        eq[1].set_color(TEAL_C)
        eq[3].set_color(TEAL_C)

        Z = VGroup(Z1,Z2,Z3,Z4).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)
        A = VGroup(A1,A2,A3,A4).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)
        A_e = VGroup(A1_e,A2_e,A3_e,A4_e).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)
        A_num = VGroup(A1_num,A2_num,A3_num,A4_num).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)
        A_sum =  MathTex(r"A = A_1 + A_2 + A_3 + A_4 ", font_size=60, color=TEAL_C).to_edge(LEFT)
        A_extra = MathTex(r"A = A_1 + A_2 + A_3 + A_4 \\+ A_5 + A_6 + A_7 + A_8", font_size=60, color=TEAL_C).to_edge(LEFT)

        
        self.play(Create(ax))
        self.wait()
        self.play(Create(graph, run_time=1))
        self.play(Write(Z, run_Time=5))
        self.play(TransformMatchingShapes(Z,A))
        self.wait()
        self.play(TransformMatchingShapes(A,A_e))
        self.wait()
        self.play(TransformMatchingShapes(A_e,A_num))
        self.wait()
        self.play(TransformMatchingShapes(A_num,A_sum))
        self.wait()
        self.play(Create(graph2_1))
        self.play(Transform(graph2_1,graph2_2))
        self.play(Transform(graph2_1,graph2_3))
        self.play(Transform(graph2_1,graph2_4))
        self.play(Transform(graph2_1,graph2))
        #self.play(Create(graph2, run_time=5))
        self.wait()
        self.play(TransformMatchingShapes(A_sum,A_extra))
        self.play(Transform(graph2_1,graph3))
        self.wait()
        self.play(Transform(A_extra,eq))
        self.wait()


class Math(Scene):
    def construct(self):
        
        za = MathTex(r"z_{j, i}^{[l]} &= \sum_k w_{j, k}^{[l]} a_{k, i}^{[l - 1]} + b_j^{[l]}, \\a_{j, i}^{[l]} &= g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}).")
        vec = MathTex(r"""
        \begin{bmatrix}
        z_{1, i}^{[l]} \\
        \vdots \\
        z_{j, i}^{[l]} \\
        \vdots \\
        z_{n^{[l]}, i}^{[l]}
        \end{bmatrix} &=
        \begin{bmatrix}
        w_{1, 1}^{[l]} & \dots & w_{1, k}^{[l]} & \dots & w_{1, n^{[l - 1]}}^{[l]} \\
        \vdots & \ddots & \vdots & \ddots & \vdots \\
        w_{j, 1}^{[l]} & \dots & w_{j, k}^{[l]} & \dots & w_{j, n^{[l - 1]}}^{[l]} \\
        \vdots & \ddots & \vdots & \ddots & \vdots \\
        w_{n^{[l]}, 1}^{[l]} & \dots & w_{n^{[l]}, k}^{[l]} & \dots & w_{n^{[l]}, n^{[l - 1]}}^{[l]}
        \end{bmatrix}
        \begin{bmatrix}
        a_{1, i}^{[l - 1]} \\
        \vdots \\
        a_{k, i}^{[l - 1]} \\
        \vdots \\
        a_{n^{[l - 1]}, i}^{[l - 1]}
        \end{bmatrix} +
        \begin{bmatrix}
        b_1^{[l]} \\
        \vdots \\
        b_j^{[l]} \\
        \vdots \\
        b_{n^{[l]}}^{[l]}
        \end{bmatrix}, \\
        \begin{bmatrix}
        a_{1, i}^{[l]} \\
        \vdots \\
        a_{j, i}^{[l]} \\
        \vdots \\
        a_{n^{[l]}, i}^{[l]}
        \end{bmatrix} &=
        \begin{bmatrix}
        g_1^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
        \vdots \\
        g_j^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
        \vdots \\
        g_{n^{[l]}}^{[l]}(z_{1, i}^{[l]}, \dots, z_{j, i}^{[l]}, \dots, z_{n^{[l]}, i}^{[l]}) \\
        \end{bmatrix}.
        """, font_size=40)

        backwardderv = MathTex(r"""
                \frac{\partial J}{\partial w_{j, k}^{[l]}} &= \sum_i \frac{\partial J}{\partial z_{j, i}^{[l]}} \frac{\partial z_{j, i}^{[l]}}{\partial w_{j, k}^{[l]}} = \sum_i \frac {\partial J}{\partial z_{j, i}^{[l]}} a_{k, i}^{[l - 1]}, \\
                \frac{\partial J}{\partial b_j^{[l]}} &= \sum_i \frac{\partial J}{\partial z_{j, i}^{[l]}} \frac{\partial z_{j, i}^{[l]}}{\partial b_j^{[l]}} = \sum_i \frac{\partial J}{\partial z_{j, i}^{[l]}}. """,
                font_size=50)

        
        zd = MathTex(r"""
\frac{\partial J}{\partial z_{j, i}^{[l]}} = \sum_p \frac{\partial J}{\partial a_{p, i}^{[l]}} \frac{\partial a_{p, i}^{[l]}}{\partial z_{j, i}^{[l]}},
""", font_size=60)


        self.play(Write(za, run_time=5))
        self.wait()
        self.play(FadeOut(za, run_time=2))
        self.play(Write(vec, run_time=8))
        self.wait()
        self.play(FadeOut(vec, run_time=2))
        self.wait()
        self.play(Write(backwardderv, run_time=5))
        self.wait()
        self.play(FadeOut(backwardderv, run_time=2))
        self.play(Write(zd), run_time=3)
        self.wait()

