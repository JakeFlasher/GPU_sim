# SM86 no-fence store-feedback qualification — H68 negative — 2026-10-16

The existing 30-cell `store_ack_frontier` matrix shows a real no-fence store
completion gap (for example, 84-block depth-16 no-fence stores are ~3,500 GPC
cycles above control and depth-64 are ~11,400 cycles above control), but the
current simulator candidate does not gate `EXIT` retirement on those acks.

As a bounded qualification experiment, the H19 feedback candidate was enabled
for every static global STORE/RED kernel rather than only device-fence/ERRBAR
shapes. Five real full replay roots completed `ok` (20 launches), but all
GPC/SM-active/frontier predictions were bit-for-bit unchanged and admission
queue exposure remained zero. The candidate alone therefore cannot model the
implicit CTA-EXIT drain; enabling it globally without an EXIT gate would be a
label-only change. The qualification source was restored.

The actual next timing seam remains a typed EXIT-drain arm, requiring an
independent store/no-fence versus explicit-fence probe and rollback against the
H41 baseline. No default timing behavior changed.
