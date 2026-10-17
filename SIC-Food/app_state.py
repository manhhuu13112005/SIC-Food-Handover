"""Session-only inputs/results; widget cleanup on navigation cannot erase them."""
import hashlib
import streamlit as st


def reset_results(namespace):
    st.session_state[namespace + '_items'] = []
    st.session_state[namespace + '_edits'] = {}
    for suffix in ('result', 'annotated', 'model_names', 'elapsed'):
        st.session_state.pop(namespace + '_' + suffix, None)
    # Widget keys never hold the only copy of the user's edits.
    for key in list(st.session_state):
        if key.startswith(namespace + '_field_'):
            del st.session_state[key]


def accept_upload(namespace, uploaded):
    payload = uploaded.getvalue() if uploaded is not None else None
    token = hashlib.sha256(payload).hexdigest() if payload else None
    if token != st.session_state.get(namespace + '_token'):
        reset_results(namespace)
        st.session_state[namespace + '_token'] = token
        st.session_state[namespace + '_input'] = payload
        st.session_state[namespace + '_filename'] = getattr(uploaded, 'name', 'Ảnh đã tải')


def _upload_changed(namespace, key):
    accept_upload(namespace, st.session_state.get(key))


def upload(namespace, label, types):
    epoch = st.session_state.get(namespace + '_upload_epoch', 0)
    key = f'{namespace}_upload_{epoch}'
    uploaded = st.file_uploader(label, type=types, key=key, max_upload_size=64 if namespace == 'vid' else 20,
                                on_change=_upload_changed, args=(namespace, key))
    # Also permits AppTest's explicit upload fixture (which has no widget callback).
    if uploaded is not None:
        accept_upload(namespace, uploaded)
    return st.session_state.get(namespace + '_input')


def clear_input(namespace):
    accept_upload(namespace, None)
    st.session_state[namespace + '_upload_epoch'] = st.session_state.get(namespace + '_upload_epoch', 0) + 1


def save_edit(namespace, item_id, field, key):
    edits = st.session_state.setdefault(namespace + '_edits', {})
    values = edits.setdefault(item_id, {})
    value = st.session_state[key]
    if field in ('ingredient', 'prep', 'condition') and values.get(field) != value:
        values.update(confirmed=False, accept=False, profile='')
        for reset in ('confirmed', 'accept', 'profile'):
            st.session_state.pop(f'{namespace}_field_{item_id}_{reset}', None)
        if field == 'condition':
            from state_policy import EXCLUDED_CONDITIONS
            values['include'] = value not in EXCLUDED_CONDITIONS
            st.session_state.pop(f'{namespace}_field_{item_id}_include', None)
    values[field] = value


def edit_widget(namespace, item_id, field, factory, label, default, options=None, **kwargs):
    values = st.session_state.get(namespace + '_edits', {}).get(item_id, {})
    value = values.get(field, default)
    key = f'{namespace}_field_{item_id}_{field}'
    common = dict(key=key, on_change=save_edit, args=(namespace, item_id, field, key), **kwargs)
    if options is not None:
        if value not in options:
            value = options[0]
        return factory(label, options, index=options.index(value), **common)
    return factory(label, value=value, **common)

