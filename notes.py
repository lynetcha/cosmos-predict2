# Github release steps


# Convert DCP checkpoint to .pt
mkdir -p checkpoints/Cosmos-Predict2-2B-TextVideo2World-Multiview
aws s3 sync --profile s3-dir s3://checkpoints-us-east-1/cosmos_predict2_multiview/cosmos2_mv/xiaomi_predict2_2b_vid2vid_mv_7views_res720_fps10_t8_fromPre32k_alpamayo2tar_2p83s_noviewprefix_1cap_cond012-0/checkpoints/iter_000003500/ checkpoints/Cosmos-Predict2-2B-TextVideo2World-Multiview


# TODO: Natten integration

# Re-enable guardrail and prompt refiner
