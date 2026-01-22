import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import plotly.graph_objects as go

class CorporateViz:
    def __init__(self, palette=None, bg_color='#FAFAFA', font_family='sans-serif'):
        """
        Initialize the corporate visualization style.
        """
        self.palette = palette if palette else ['#0F294A', '#1B9CFC', '#5D6D7E', '#AED6F1', '#D4E6F1']
        self.bg_color = bg_color
        self.font = font_family
        
        plt.rcParams['font.family'] = self.font
        plt.rcParams['text.color'] = self.palette[0]
        plt.rcParams['axes.labelcolor'] = self.palette[0]
        plt.rcParams['xtick.color'] = self.palette[0]
        plt.rcParams['ytick.color'] = self.palette[0]

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------
    def _get_colors(self, n, custom_colors=None):
        if custom_colors: return custom_colors
        return [self.palette[i % len(self.palette)] for i in range(n)]

    def _setup_axis(self, ax, grid='y'):
        """Cleans spines and sets grid."""
        ax.set_facecolor(self.bg_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if grid == 'y':
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('none') 
            ax.grid(axis='y', color='#EEEEEE', linewidth=1, zorder=0)
        elif grid == 'x':
            ax.spines['bottom'].set_color('#DDDDDD')
            ax.xaxis.set_ticks_position('none')
            ax.grid(axis='x', color='#EEEEEE', linewidth=1, zorder=0)
        return ax

    def _apply_gradient(self, ax, bars, orientation='vertical'):
        """Applies a gradient that matches the color of EACH bar."""
        gradient_data = np.atleast_2d(np.linspace(0, 1, 256))
        
        for bar in bars:
            base_color = bar.get_facecolor() 
            bar.set_facecolor("none") 
            light_color = [x + (1 - x) * 0.5 for x in base_color[:3]] 
            cmap = mcolors.LinearSegmentedColormap.from_list("grad", [base_color, light_color])
            
            x, y = bar.get_x(), bar.get_y()
            w, h = bar.get_width(), bar.get_height()
            
            if orientation == 'vertical':
                ax.imshow(gradient_data.T, extent=[x, x+w, y, y+h], aspect='auto', cmap=cmap, zorder=3)
            else:
                ax.imshow(gradient_data, extent=[x, x+w, y, y+h], aspect='auto', cmap=cmap, zorder=3)
            ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor=base_color, lw=0.5, zorder=4))

    def _style_axis_labels(self, ax, axis='x', font_dict=None, style_key='axis_label'):
        """Robustly applies font settings to axis ticks."""
        fd = font_dict or {}
        style = fd.get(style_key, {})
        if not style: return

        tick_params = {}
        if 'fontsize' in style: tick_params['labelsize'] = style['fontsize']
        if 'color' in style: tick_params['labelcolor'] = style['color']
        if 'rotation' in style: tick_params['rotation'] = style['rotation']
        
        if axis == 'x': ax.tick_params(axis='x', **tick_params)
        else: ax.tick_params(axis='y', **tick_params)

        if 'fontweight' in style or 'family' in style:
            plt.draw() 
            labels = ax.get_xticklabels() if axis == 'x' else ax.get_yticklabels()
            for label in labels:
                if 'fontweight' in style: label.set_fontweight(style['fontweight'])
                if 'family' in style: label.set_family(style['family'])

    def _apply_formatter(self, ax, axis, formatter):
        """Formats the numbers on the axis."""
        ax_obj = ax.xaxis if axis == 'x' else ax.yaxis
        if formatter:
            if isinstance(formatter, str):
                ax_obj.set_major_formatter(ticker.StrMethodFormatter(formatter))
            elif callable(formatter):
                ax_obj.set_major_formatter(ticker.FuncFormatter(formatter))

    def _apply_titles(self, ax, title, subtitle, font_dict):
        """Applies Title/Subtitle with optional custom positioning."""
        fd = font_dict or {}
        t_style = {'fontsize': 16, 'fontweight': 'bold', 'color': self.palette[0]}
        s_style = {'fontsize': 11, 'color': '#555555'}
        t_style.update(fd.get('title', {}))
        s_style.update(fd.get('subtitle', {}))

        t_y = t_style.pop('y', 1.12) 
        s_y = s_style.pop('y', 1.06) 
        ax.text(x=0, y=t_y, s=title, transform=ax.transAxes, ha='left', va='top', **t_style)
        if subtitle:
            ax.text(x=0, y=s_y, s=subtitle, transform=ax.transAxes, ha='left', va='top', **s_style)

    def _add_arrow_annotation(self, ax, note):
        """Draws a polished arrow pointer."""
        ax.annotate(
            note['text'], xy=(note['x'], note['y']), xytext=note['offset'],
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.2", color='#333', lw=1.5),
            fontsize=10, color='#333', backgroundcolor=self.bg_color,
            bbox=dict(pad=2, facecolor=self.bg_color, edgecolor='none', alpha=0.8)
        )

    # -------------------------------------------------------------------------
    # CHART METHODS
    # -------------------------------------------------------------------------

    def plot_bar(self, df, x_col, y_col, title, subtitle=None, 
                 custom_colors=None, font_dict=None, gradient=False, 
                 value_formatter=None, custom_labels=None, show_axis_scale=True, figsize=(10, 6)):
        """Vertical Bar Chart."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
        ax = self._setup_axis(ax, grid='y')
        
        colors = self._get_colors(len(df), custom_colors)
        bars = ax.bar(df[x_col], df[y_col], color=colors, zorder=3, width=0.6)
        
        if gradient: self._apply_gradient(ax, bars, 'vertical')
        
        if not show_axis_scale:
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(False, axis='y')
        
        self._style_axis_labels(ax, axis='x', font_dict=font_dict, style_key='axis_label') 
        if show_axis_scale:
            self._style_axis_labels(ax, axis='y', font_dict=font_dict, style_key='value_axis_style') 
            self._apply_formatter(ax, axis='y', formatter=value_formatter)

        fd = font_dict or {}
        val_style = {'fontsize': 9, 'color': self.palette[0], 'ha': 'center'}
        val_style.update(fd.get('value_label', {}))
        
        if custom_labels is not None:
            labels_to_use = custom_labels
        else:
            labels_to_use = [None] * len(bars)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            if labels_to_use[i] is not None:
                label_text = str(labels_to_use[i])
            else:
                label_text = f'{height:,.0f}'
                if value_formatter and isinstance(value_formatter, str):
                    try: label_text = value_formatter.format(x=height)
                    except: pass
            
            ax.text(bar.get_x() + bar.get_width()/2., height + (df[y_col].max()*0.01),
                    label_text, va='bottom', **val_style)

        self._apply_titles(ax, title, subtitle, font_dict)
        plt.tight_layout()
        return fig, ax

    def plot_barh(self, df, x_col, y_col, title, subtitle=None, 
                  custom_colors=None, font_dict=None, gradient=False, 
                  value_formatter=None, custom_labels=None, show_axis_scale=True, figsize=(10, 6)):
        """Horizontal Bar Chart."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
        ax = self._setup_axis(ax, grid='x')
        
        colors = self._get_colors(len(df), custom_colors)
        bars = ax.barh(df[y_col], df[x_col], color=colors, zorder=3, height=0.6)
        
        if gradient: self._apply_gradient(ax, bars, 'horizontal')
        
        if not show_axis_scale:
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False, axis='x')

        self._style_axis_labels(ax, axis='y', font_dict=font_dict, style_key='axis_label') 
        if show_axis_scale:
            self._style_axis_labels(ax, axis='x', font_dict=font_dict, style_key='value_axis_style') 
            self._apply_formatter(ax, axis='x', formatter=value_formatter)

        fd = font_dict or {}
        val_style = {'fontsize': 9, 'color': self.palette[0], 'va': 'center'}
        val_style.update(fd.get('value_label', {}))

        if custom_labels is not None:
            labels_to_use = custom_labels
        else:
            labels_to_use = [None] * len(bars)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            if labels_to_use[i] is not None:
                label_text = str(labels_to_use[i])
            else:
                label_text = f'{width:,.0f}'
                if value_formatter and isinstance(value_formatter, str):
                    try: label_text = value_formatter.format(x=width)
                    except: pass

            ax.text(width + (df[x_col].max() * 0.02), bar.get_y() + bar.get_height()/2, 
                    label_text, **val_style)

        self._apply_titles(ax, title, subtitle, font_dict)
        plt.tight_layout()
        return fig, ax

    def plot_stacked_bar(self, df, x_col, stack_cols, title, subtitle=None, 
                         font_dict=None, show_axis_scale=True, value_formatter=None, 
                         show_total_labels=True, figsize=(10, 6)):
        """
        Vertical Stacked Bar. 
        :param show_total_labels: Boolean. If True, shows the Sum at the top of the stack.
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
        ax = self._setup_axis(ax, grid='y')
        
        colors = self._get_colors(len(stack_cols))
        x_data = df[x_col]
        bottom_vals = np.zeros(len(df))
        
        for i, col in enumerate(stack_cols):
            ax.bar(x_data, df[col], bottom=bottom_vals, label=col, 
                   color=colors[i], zorder=3, width=0.6)
            bottom_vals += df[col].values

        if not show_axis_scale:
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(False, axis='y')

        self._style_axis_labels(ax, axis='x', font_dict=font_dict, style_key='axis_label') 
        if show_axis_scale:
            self._style_axis_labels(ax, axis='y', font_dict=font_dict, style_key='value_axis_style') 
            self._apply_formatter(ax, axis='y', formatter=value_formatter)

        # Draw Totals ONLY if requested
        if show_total_labels:
            fd = font_dict or {}
            total_style = {'fontsize': 9, 'fontweight': 'bold', 'color': self.palette[0]}
            total_style.update(fd.get('value_label', {}))
            
            for x, total in zip(x_data, bottom_vals):
                label_text = f'{total:,.0f}'
                if value_formatter and isinstance(value_formatter, str):
                    try: label_text = value_formatter.format(x=total)
                    except: pass
                
                ax.text(x, total + (max(bottom_vals)*0.01), label_text, 
                        ha='center', va='bottom', **total_style)

        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1), ncol=len(stack_cols))
        self._apply_titles(ax, title, subtitle, font_dict)
        ax.texts[0].set_position((0, 1.15)) 
        if subtitle: ax.texts[1].set_position((0, 1.10))
        plt.tight_layout()
        return fig, ax

    def plot_stacked_barh(self, df, y_col, stack_cols, title, subtitle=None, 
                          font_dict=None, show_axis_scale=True, value_formatter=None, 
                          show_total_labels=True, figsize=(10, 6)):
        """
        Horizontal Stacked Bar.
        :param show_total_labels: Boolean. If True, shows the Sum at the end of the stack.
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
        ax = self._setup_axis(ax, grid='x')
        
        colors = self._get_colors(len(stack_cols))
        y_data = df[y_col]
        left_vals = np.zeros(len(df))
        
        for i, col in enumerate(stack_cols):
            ax.barh(y_data, df[col], left=left_vals, label=col, 
                    color=colors[i], zorder=3, height=0.6)
            left_vals += df[col].values

        if not show_axis_scale:
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(False, axis='x')

        self._style_axis_labels(ax, axis='y', font_dict=font_dict, style_key='axis_label') 
        if show_axis_scale:
            self._style_axis_labels(ax, axis='x', font_dict=font_dict, style_key='value_axis_style') 
            self._apply_formatter(ax, axis='x', formatter=value_formatter)

        if show_total_labels:
            fd = font_dict or {}
            total_style = {'fontsize': 9, 'fontweight': 'bold', 'color': self.palette[0]}
            total_style.update(fd.get('value_label', {}))
            
            for y, total in zip(y_data, left_vals):
                label_text = f'{total:,.0f}'
                if value_formatter and isinstance(value_formatter, str):
                    try: label_text = value_formatter.format(x=total)
                    except: pass
                    
                ax.text(total + (max(left_vals)*0.01), y, label_text, 
                        ha='left', va='center', **total_style)

        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1), ncol=len(stack_cols))
        self._apply_titles(ax, title, subtitle, font_dict)
        ax.texts[0].set_position((0, 1.15)) 
        if subtitle: ax.texts[1].set_position((0, 1.10))
        plt.tight_layout()
        return fig, ax

    def plot_timeseries(self, df, date_col, value_col, title, subtitle=None, 
                        annotations=None, gradient=False, figsize=(12, 6), font_dict=None):
        """Standard Time Series."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
        ax = self._setup_axis(ax, grid='y')
        
        df[date_col] = pd.to_datetime(df[date_col])
        x, y = df[date_col].values, df[value_col].values
        
        ax.plot(x, y, color=self.palette[1], linewidth=2.5, zorder=3)
        
        if gradient:
            for i in range(20):
                ax.fill_between(x, y, y2=0, color=self.palette[1], alpha=0.03*(1-i/20), zorder=2)
        else:
            ax.fill_between(x, y, color=self.palette[1], alpha=0.1, zorder=2)
            
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        self._apply_titles(ax, title, subtitle, font_dict)
        if annotations:
            for note in annotations: self._add_arrow_annotation(ax, note)
        plt.tight_layout()
        return fig, ax

    def plot_dual_timeseries(self, df, date_col, col1, col2, title, subtitle=None, 
                             annotations=None, figsize=(12, 6), font_dict=None):
        """Comparing two time series."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
        ax = self._setup_axis(ax, grid='y')
        df[date_col] = pd.to_datetime(df[date_col])
        
        ax.plot(df[date_col], df[col1], color=self.palette[1], linewidth=3, label=col1, zorder=3)
        ax.plot(df[date_col], df[col2], color='#95A5A6', linewidth=2, linestyle='--', label=col2, zorder=3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.legend(frameon=False, loc='upper right')
        self._apply_titles(ax, title, subtitle, font_dict)
        if annotations:
            for note in annotations: self._add_arrow_annotation(ax, note)
        plt.tight_layout()
        return fig, ax

    def plot_donut(self, df, cat_col, val_col, title, subtitle=None, 
                   font_dict=None, start_angle=90, show_center_label=True, figsize=(10, 6)):
        """Donut chart with callouts."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
        
        colors = self._get_colors(len(df))
        data = df[val_col].values
        labels = df[cat_col].values
        
        wedges, texts = ax.pie(
            data, startangle=start_angle, colors=colors, center=(0, -0.1), 
            wedgeprops=dict(width=0.5, edgecolor=self.bg_color),
            textprops=dict(color=self.palette[0])
        )
        
        fd = font_dict or {}
        label_style = {'fontsize': 10, 'fontweight': 'normal'} 
        label_style.update(fd.get('value_label', {}))
        user_color = label_style.get('color', None)
        if 'color' in label_style: del label_style['color']
        
        bbox_props = dict(boxstyle="square,pad=0.3", fc=self.bg_color, ec="none", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-", linewidth=1.5), bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang)) - 0.1 
            x = np.cos(np.deg2rad(ang))
            
            percent = data[i] / data.sum() * 100
            label_txt = f"{labels[i]}\n{percent:.1f}%"
            
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            
            kw["arrowprops"].update({"color": colors[i], "connectionstyle": connectionstyle})
            final_text_color = user_color if user_color else colors[i]
            
            ax.annotate(label_txt, xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, **kw,
                        color=final_text_color, **label_style)

        if show_center_label:
            total_val = f"{data.sum():,.0f}"
            ax.text(0, -0.1, f"Total\n{total_val}", ha='center', va='center', 
                    fontsize=12, fontweight='bold', color=self.palette[0])

        self._apply_titles(ax, title, subtitle, fd)
        plt.tight_layout()
        return fig, ax

    def plot_sankey(self, labels, source, target, value, title, static_file=None):
        """Sankey via Plotly."""
        hex_c = self.palette[1].lstrip('#')
        rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        link_color = f"rgba{rgb + (0.3,)}"
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=self.palette),
            link=dict(source=source, target=target, value=value, color=link_color)
        )])
        fig.update_layout(title_text=f"<b>{title}</b>", font_family=self.font, 
                          paper_bgcolor=self.bg_color, plot_bgcolor=self.bg_color, font_size=12)
        
        if static_file:
            fig.write_image(static_file, scale=3)
        else:
            return fig