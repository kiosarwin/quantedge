# Data Schema Quant Mini

> Tujuan schema ini adalah memastikan satu trade dapat dijelaskan ulang dari data, tanpa bergantung pada state runtime.

## 1) Identitas Trade

| Field | Tipe | Sumber | Wajib | Catatan |
|---|---|---:|---:|---|
| `trade_id` | string | system | ya | Identifier unik trade |
| `symbol` | string | market data | ya | Pair yang diperdagangkan |
| `direction` | string | signal | ya | `long` / `short` |
| `open_ts` | float/int | execution | ya | Waktu buka |
| `close_ts` | float/int | execution | ya | Waktu tutup |

## 2) Konteks Strategi

| Field | Tipe | Sumber | Wajib | Catatan |
|---|---|---:|---:|---|
| `regime` | string | scorer | ya | State market utama |
| `sleeve` | string | router | ya | Sub-strategi yang dipilih |
| `exit_profile` | string | risk manager | ya | Profile exit yang dipakai |
| `signal_type` | string | scorer | ya | Jenis setup |
| `entry_reason` | string | scorer | ya | Alasan entry |
| `cohort_key` | string | cohort policy | tidak | Grup setup apabila tersedia |

## 3) Skor dan Fitur

| Field | Tipe | Sumber | Wajib | Catatan |
|---|---|---:|---:|---|
| `total_score` | float | scorer | ya | Score akhir |
| `trend_strength` | float | scorer | ya | Komponen sinyal |
| `volume_confirmation` | float | scorer | ya | Komponen sinyal |
| `structure_quality` | float | scorer | ya | Komponen sinyal |
| `open_interest` | float | scorer | ya | Komponen sinyal |
| `funding_sentiment` | float | scorer | ya | Komponen sinyal |
| `order_book` | float | scorer | ya | Komponen sinyal |
| `volatility` | float | scorer | ya | Komponen sinyal |
| `dispersion_value` | float | scorer | ya | Overlay risiko |
| `dispersion_state` | string | scorer | ya | `normal`, `warn`, `high` |

## 4) EV / ML / Risk

| Field | Tipe | Sumber | Wajib | Catatan |
|---|---|---:|---:|---|
| `ev_p_win` | float | EV model | ya | Probabilitas menang versi EV |
| `ev_net_pct` | float | EV model | ya | EV bersih |
| `ev_confidence` | float | EV model | ya | Confidence EV |
| `ml_p_win` | float | ML | ya | Probabilitas menang versi ML |
| `ml_threshold` | float | ML | ya | Ambang veto ML |
| `ml_ready` | bool | ML | ya | Model siap gate atau tidak |
| `veto_reason` | string | system | tidak | Alasan reject bila ada |
| `size_multiplier_breakdown` | object/string | fund/risk/ML | ya | Jejak sizing |

## 5) Outcome

| Field | Tipe | Sumber | Wajib | Catatan |
|---|---|---:|---:|---|
| `entry_price` | float | execution | ya | Harga masuk |
| `exit_price` | float | execution | ya | Harga keluar |
| `pnl_usd` | float | execution | ya | PnL nominal |
| `pnl_pct` | float | execution | ya | PnL persen |
| `mfe_r` | float | trade manager | ya | Max favorable excursion |
| `mae_r` | float | trade manager | ya | Max adverse excursion |
| `hold_duration_s` | float | trade manager | ya | Lama hold |

## 6) Label Tambahan untuk ML

Gunakan label berikut untuk training dan audit:

- `is_real`
- `is_shadow`
- `is_paper`
- `was_vetoed_by_ml`
- `was_vetoed_by_risk`
- `was_vetoed_by_fm`
- `was_high_conviction`

## 7) Aturan Praktis

- Jangan gabungkan real dan shadow tanpa flag.
- Jangan hilangkan `regime`, `sleeve`, dan `exit_profile`.
- Jangan latih model tanpa urutan timestamp yang benar.
- Jangan evaluasi model tanpa memisahkan in-sample dan out-of-sample.
- Semua field yang dipakai untuk keputusan harus dapat direkonstruksi dari log.
