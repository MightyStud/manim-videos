from manim import *
import math 

#changes according to scene
config.background_color = BLACK

class nneq(Scene):
    def construct(self):
        eq = MathTex(r"WX + b", font_size=350)
        eq.set_color(BLACK)
        self.play(Write(eq, run_time=3.5))


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
        graph = ax.plot(lambda x: - x**3 - 2 * x**2 + 2 * x + 1, x_range=[-4,4], color=MAROON_C)
        graph2 = ax.plot(lambda x:  4.492*x +3.07 , x_range=[-4,4], color=TEAL_C)
        

        Z1 = MathTex(r"Z_1 = W_1x + b_1", font_size=50, color=MAROON_C)
        Z2 = MathTex(r"Z_2 = W_2x + b_2", font_size=50, color=MAROON_C)
        Z3 = MathTex(r"Z_3 = W_3x + b_3", font_size=50, color=MAROON_C)
        Z4 = MathTex(r"Z_4 = W_4x + b_4", font_size=50, color=MAROON_C)

        Z1_n = MathTex(r"Z_1 = n_1 (W_1x + b_1)", font_size=50, color=MAROON_C)
        Z2_n = MathTex(r"Z_2 = n_2 (W_2x + b_2)", font_size=50, color=MAROON_C)
        Z3_n = MathTex(r"Z_3 = n_3 (W_3x + b_3)", font_size=50, color=MAROON_C)
        Z4_n = MathTex(r"Z_4 = n_4 (W_4x + b_4)", font_size=50, color=MAROON_C)

        Z1_num = MathTex(r"Z_1 = 3.26 (1.08x -3.17)", font_size=50, color=MAROON_C)
        Z2_num = MathTex(r"Z_2 = -2.44 (3.1x -3.83)", font_size=50, color=MAROON_C)
        Z3_num = MathTex(r"Z_3 = -2.43 (-1.42x + 2.74)", font_size=50, color=MAROON_C)
        Z4_num = MathTex(r"Z_4 = 3.39 (1.5x + 3.66)", font_size=50, color=MAROON_C)
    

        Z = VGroup(Z1, Z2, Z3, Z4).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)
        Z_n = VGroup(Z1_n, Z2_n, Z3_n, Z4_n).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)
        Z_num = VGroup(Z1_num, Z2_num, Z3_num, Z4_num).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)
        Z_num = VGroup(Z1_num, Z2_num, Z3_num, Z4_num).arrange(DOWN, center=False, aligned_edge=LEFT).to_edge(UL).shift(DOWN*1.9)

        Z_sum =  MathTex(r"Z = Z_1 + Z_2 + Z_3 + Z_4 + b", font_size=50, color=TEAL_C).next_to(Z_num,DOWN)
        Z_sumed = MathTex(r"Z = 4.492x + 3.07", font_size=90, color=TEAL_C).to_edge(LEFT)
    

        self.play(Create(ax))
        self.play(Create(graph),)
        self.play(Write(Z), run_time=2)
        self.wait()
        self.play(Transform(Z, Z_n))
        self.wait()
        self.play(Transform(Z, Z_num))
        self.wait()
        self.play(Write(Z_sum))
        self.wait()
        self.play(AnimationGroup(Transform(Z_sum, Z_sumed), FadeOut(Z) , Create(graph2, run_time=2)))
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
        relu0 = MathTex(r"f(x) = Rectified Linear Unit")
        relu1 =  MathTex(r"f(x) = ReLU(x)")
        relu2 =  MathTex(r"f(x) = ReLU(x)")
        relu3 =  MathTex(r"f(x) = max(x,0)")
        relu4 =  MathTex(r"f(x) = max(Wx + b,0)")
        
        
        
















