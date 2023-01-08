from manim import *
import math 

config.background_color = WHITE

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


        label = MathTex(r"f(x) &= Wx + b \\ W &= \\ b &=", font_size=50, color=MAROON_C, substrings_to_isolate="x").to_edge(UL)
        label.set_color_by_tex('x', TEAL_C)

        
        graph = FunctionGraph(lambda x: W.get_value() * x + b.get_value(), color=MAROON_C)
        graph.add_updater(lambda func: func.become(FunctionGraph(lambda x: W.get_value() * x + b.get_value(), color=MAROON_C)))

        self.play(Write(ax))
        self.play(AnimationGroup(Write(label), Write(W_num), Write(b_num), Write(graph)))
        self.wait()
        self.play(W.animate.set_value(5), run_time=1)
        self.play(W.animate.set_value(-5), run_time=2)
        self.play(W.animate.set_value(1), run_time=1)
        self.play(b.animate.set_value(3), run_time=1)
        self.play(b.animate.set_value(-3), run_time=2)
        self.play(AnimationGroup(b.animate.set_value(0)), run_time=2)
        self.wait()









