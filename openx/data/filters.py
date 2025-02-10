import tensorflow as tf


def filter_by_ep_path(search_strings, path_key: str = "file_path"):
    def _filter(ep):
        string = ep["episode_metadata"][path_key]
        if isinstance(search_strings, list):
            bool_tensor = tf.concat(
                [tf.strings.regex_full_match(string, pattern=".*" + s + ".*") for s in search_strings], axis=0
            )
            return tf.math.reduce_all(bool_tensor)
        return tf.strings.regex_full_match(string, pattern=".*" + search_strings + ".*")

    return _filter


def quality_filter(threshold):
    def _filter(ep):
        return tf.cast(ep["episode_metadata"]["quality_score"] >= threshold, tf.bool)

    return _filter
