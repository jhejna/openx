from typing import Dict

import tensorflow as tf

from openx.data.utils import RobotType, StateEncoding

LIBERO_TASK_IDS = {
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet": 40,
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it": 41,
    "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet": 42,
    "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it": 43,
    "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it": 44,
    "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it": 45,
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet": 46,
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet": 47,
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it": 48,
    "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate": 49,
    "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet": 50,
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet": 51,
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate": 52,
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate": 53,
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate": 54,
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet": 55,
    "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle": 56,
    "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl": 57,
    "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove": 58,
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove": 59,
    "KITCHEN_SCENE3_turn_on_the_stove": 60,
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it": 61,
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": 32,
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet": 62,
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer": 63,
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet": 64,
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": 33,
    "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet": 65,
    "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet": 66,
    "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack": 67,
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet": 68,
    "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet": 69,
    "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate": 70,
    "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet": 71,
    "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet": 72,
    "KITCHEN_SCENE6_close_the_microwave": 73,
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": 39,
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug": 74,
    "KITCHEN_SCENE7_open_the_microwave": 75,
    "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate": 76,
    "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate": 77,
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": 38,
    "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove": 78,
    "KITCHEN_SCENE8_turn_off_the_stove": 79,
    "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf": 80,
    "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet": 81,
    "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf": 82,
    "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet": 83,
    "KITCHEN_SCENE9_turn_on_the_stove": 84,
    "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it": 85,
    "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket": 86,
    "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket": 87,
    "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket": 88,
    "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket": 89,
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": 37,
    "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket": 90,
    "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket": 91,
    "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket": 92,
    "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket": 93,
    "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket": 94,
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": 30,
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": 31,
    "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray": 95,
    "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray": 96,
    "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray": 97,
    "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray": 98,
    "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray": 99,
    "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray": 100,
    "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray": 101,
    "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray": 102,
    "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray": 103,
    "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray": 104,
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate": 105,
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate": 106,
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate": 107,
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": 34,
    "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate": 108,
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate": 109,
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate": 110,
    "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate": 111,
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate": 112,
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": 36,
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": 35,
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy": 113,
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy": 114,
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy": 115,
    "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy": 116,
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": 117,
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy": 118,
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy": 119,
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy": 120,
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy": 121,
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy": 122,
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy": 123,
    "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy": 124,
    "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy": 125,
    "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf": 126,
    "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf": 127,
    "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf": 128,
    "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf": 129,
    "open_the_middle_drawer_of_the_cabinet": 20,
    "open_the_top_drawer_and_put_the_bowl_inside": 23,
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket": 10,
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket": 13,
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": 0,
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate": 2,
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate": 4,
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate": 6,
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate": 8,
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate": 1,
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate": 3,
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": 5,
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": 7,
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate": 9,
    "pick_up_the_butter_and_place_it_in_the_basket": 16,
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket": 18,
    "pick_up_the_cream_cheese_and_place_it_in_the_basket": 11,
    "pick_up_the_ketchup_and_place_it_in_the_basket": 14,
    "pick_up_the_milk_and_place_it_in_the_basket": 17,
    "pick_up_the_orange_juice_and_place_it_in_the_basket": 19,
    "pick_up_the_salad_dressing_and_place_it_in_the_basket": 12,
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket": 15,
    "push_the_plate_to_the_front_of_the_stove": 25,
    "put_the_bowl_on_the_plate": 28,
    "put_the_bowl_on_the_stove": 21,
    "put_the_bowl_on_top_of_the_cabinet": 24,
    "put_the_cream_cheese_in_the_bowl": 26,
    "put_the_wine_bottle_on_the_rack": 29,
    "put_the_wine_bottle_on_top_of_the_cabinet": 22,
    "turn_on_the_stove": 27,
}

LIBERO_TASK_ID_LOOKUP_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(LIBERO_TASK_IDS.keys())),
        values=tf.constant(list(LIBERO_TASK_IDS.values()), dtype=tf.int32),
    ),
    default_value=-1,
)


def libero_dataset_transform(ep: Dict):
    # Libero Only: Parse task IDs
    task = ep["episode_metadata"]["task"]
    task_id = tf.repeat(LIBERO_TASK_ID_LOOKUP_TABLE.lookup(task[0]), tf.shape(tf.nest.flatten(ep)[0])[0])

    observation = {
        "image": {"agent": ep["observation"]["agent_image"], "wrist": ep["observation"]["wrist_image"]},
        "state": {
            StateEncoding.EE_POS: ep["observation"]["state"]["ee_pos"],
            StateEncoding.EE_EULER: ep["observation"]["state"]["ee_euler"],
            StateEncoding.GRIPPER: ep["observation"]["state"]["gripper_qpos"][..., :1],
            StateEncoding.JOINT_POS: ep["observation"]["state"]["joint_pos"],
        },
        "language_instruction": ep["language_instruction"],
        "task_id": task_id,
    }

    action = {
        "desired_delta": {
            StateEncoding.EE_POS: ep["action"][..., :3],
            StateEncoding.EE_EULER: ep["action"][..., 3:6],
        },
        "desired_absolute": {
            StateEncoding.GRIPPER: ep["action"][..., -1:],
        },
    }

    ep["observation"] = observation
    ep["action"] = action
    ep["robot"] = RobotType.PANDA
    ep["ep_idx"] = ep["episode_metadata"]["ep_idx"]
    ep["demo_idx"] = ep["episode_metadata"]["demo_idx"]
    # Also write in tasks at top level.
    ep["task"] = task
    ep["task_id"] = task_id
    return ep
