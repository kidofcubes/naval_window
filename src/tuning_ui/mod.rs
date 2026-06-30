use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPrimaryContextPass};

use crate::fftwaves::{FftCascadesUniform, FftSpectrumParams, FFT_CASCADE_COUNT};
use crate::watering::WaterSettings;

pub struct TuningUiPlugin;

impl Plugin for TuningUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(EguiPrimaryContextPass, draw_tuning_panel);
    }
}

fn draw_tuning_panel(
    mut contexts: EguiContexts,
    mut water_settings: ResMut<WaterSettings>,
    mut fft_cascades: ResMut<FftCascadesUniform>,
    mut fft_spectrum: ResMut<FftSpectrumParams>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    egui::Window::new("Ocean tuning")
        .default_open(true)
        .resizable(true)
        .show(ctx, |ui| {
            egui::CollapsingHeader::new("Realwaves ζ range")
                .default_open(true)
                .show(ui, |ui| {
                    ui.add(
                        egui::Slider::new(&mut water_settings.zeta_min, -8.0..=4.0)
                            .text("zeta_min"),
                    );
                    ui.add(
                        egui::Slider::new(&mut water_settings.zeta_max, -2.0..=10.0)
                            .text("zeta_max"),
                    );
                    if water_settings.zeta_max < water_settings.zeta_min {
                        water_settings.zeta_max = water_settings.zeta_min;
                    }
                });

            egui::CollapsingHeader::new("FFT JONSWAP spectrum")
                .default_open(true)
                .show(ui, |ui| {
                    ui.add(
                        egui::Slider::new(&mut fft_spectrum.wind_speed, 0.5..=30.0)
                            .text("wind_speed (m/s)"),
                    );

                    let mut angle =
                        fft_spectrum.wind_dir_z.atan2(fft_spectrum.wind_dir_x).to_degrees();
                    if ui
                        .add(egui::Slider::new(&mut angle, -180.0..=180.0).text("wind_dir (°)"))
                        .changed()
                    {
                        let rad = angle.to_radians();
                        fft_spectrum.wind_dir_x = rad.cos();
                        fft_spectrum.wind_dir_z = rad.sin();
                    }

                    ui.add(
                        egui::Slider::new(&mut fft_spectrum.fetch, 1_000.0..=1_000_000.0)
                            .text("fetch (m)")
                            .logarithmic(true),
                    );
                    ui.add(
                        egui::Slider::new(&mut fft_spectrum.gamma, 1.0..=7.0).text("gamma"),
                    );
                    ui.add(
                        egui::Slider::new(&mut fft_spectrum.amplitude, 0.001..=5.0)
                            .text("amplitude")
                            .logarithmic(true),
                    );
                });

            egui::CollapsingHeader::new("FFT cascades")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label("Each row: (patch length, k_min, k_max)");
                    for c in 0..FFT_CASCADE_COUNT {
                        ui.separator();
                        ui.label(format!("Cascade {}", c));
                        let v = &mut fft_cascades.cascades[c];
                        ui.add(
                            egui::Slider::new(&mut v.x, 1.0..=2000.0)
                                .text("patch (m)")
                                .logarithmic(true),
                        );
                        ui.add(
                            egui::Slider::new(&mut v.y, 0.0..=10.0)
                                .text("k_min")
                                .logarithmic(true),
                        );
                        ui.add(
                            egui::Slider::new(&mut v.z, 0.001..=100.0)
                                .text("k_max")
                                .logarithmic(true),
                        );
                    }
                });
        });
}
