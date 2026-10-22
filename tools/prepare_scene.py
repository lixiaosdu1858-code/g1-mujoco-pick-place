"""Deterministic local XML transformation, not a robot model distribution.

Requires two user-supplied upstream XML files. No downloads, MuJoCo import,
physics integration, rendering, optimization, or model-parameter substitution.
Task geometry and transforms are extracted from our existing preparation code.
"""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

INPUT_SHA = {
    'robot': 'f177bbfecb2bc9e1cbae9dee875a121e4d19de7a34523bf3ba8db527edc8fa54',
    'scene': 'f9f18b6ed53688b3bfe8b0c5a2597fc9b7480cc2313b48035c32e7c006b6cb1d',
}
SOURCE_REVISION = '905111699ef28bdeb1c69c6fe8ada8405e963a48'
ARM = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
       'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
FINGERS = ['right_dex1_finger_joint_1', 'right_dex1_finger_joint_2']

def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()

def require(root, query):
    element = root.find(query)
    if element is None:
        raise ValueError('Missing expected XML element: ' + query)
    return element

def vector(text, count):
    values = [float(value) for value in text.split()]
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError('Malformed XML vector')
    return values

def source_tree(robot_bytes, scene_bytes):
    """Same merge/fix/filter operations as the original local builders."""
    robot, scene = ET.fromstring(robot_bytes), ET.fromstring(scene_bytes)
    includes = list(scene.iter('include'))
    if len(includes) != 1 or includes[0].get('file') != 'g1_29dof_with_grippers.xml':
        raise ValueError('Unexpected source include')
    require(robot, 'compiler').set('meshdir', '.')
    option = require(robot, 'option')
    for key, value in [('timestep','0.004'), ('integrator','implicitfast'), ('gravity','0 0 -9.81')]:
        option.set(key, value)
    for child in scene:
        if child.tag == 'include':
            continue
        if child.tag in {'asset','worldbody','default'}:
            require(robot, child.tag).extend(deepcopy(list(child)))
        elif robot.find(child.tag) is None:
            robot.append(deepcopy(child))
        else:
            raise ValueError('Unexpected duplicate source section: ' + child.tag)
    world = require(robot, 'worldbody')
    marker = world.find("site[@name='com_marker']")
    if marker is not None:
        world.remove(marker)
    if robot.find('keyframe') is not None:
        raise ValueError('Unexpected source keyframes')
    pelvis = require(robot, "worldbody/body[@name='pelvis']")
    floating = require(pelvis, "joint[@name='floating_base_joint']")
    if floating.get('type') != 'free':
        raise ValueError('Expected free pelvis joint')
    pelvis.remove(floating)
    removed = set()
    for body in world.iter('body'):
        for joint in list(body.findall('joint')):
            name = joint.get('name')
            if name in ARM + FINGERS:
                continue
            if joint.get('type') == 'slide':
                axis = vector(joint.get('axis','0 0 1'),3)
                pos = vector(body.get('pos','0 0 0'),3)
                body.set('pos',' '.join(str(p + .0245*a) for p,a in zip(pos,axis)))
            removed.add(name)
            body.remove(joint)
    actuators = require(robot,'actuator')
    for actuator in list(actuators):
        if actuator.get('joint') not in ARM + FINGERS:
            actuators.remove(actuator)
    sensor = robot.find('sensor')
    if sensor is not None:
        for element in list(sensor):
            if element.get('joint') in removed:
                sensor.remove(element)
    contact = robot.find('contact')
    if contact is None:
        contact = ET.SubElement(robot,'contact')
    ET.SubElement(contact,'exclude',{'body1':'torso_link','body2':'right_shoulder_pitch_link'})
    require(robot,".//site[@name='right_grasp_site']").attrib.update(size='0.005',rgba='0.1 0.8 0.6 1',group='0')
    robot.set('model','g1_fk_first_reachable_pose_preview')
    return robot

def add_task_layout(root, parameters):
    """Same frozen ABC-to-layout arithmetic; no FK search or new design."""
    A, B, C = (parameters['poses'][name] for name in ('A','B','C'))
    world = require(root,'worldbody')
    table_z = float(B['position_world_m'][2]-.025)
    cube = np.asarray(B['position_world_m']).copy(); cube[2] -= .005
    tray_xy = np.asarray(C['position_world_m'][:2])
    xmin, xmax = min(cube[0]-.07,tray_xy[0]-.105), max(cube[0]+.07,tray_xy[0]+.105)
    ymin, ymax = min(cube[1]-.07,tray_xy[1]-.105), max(cube[1]+.07,tray_xy[1]+.105)
    center = np.array([(xmin+xmax)/2,(ymin+ymax)/2])
    half = np.array([(xmax-xmin)/2,(ymax-ymin)/2])
    def geom(parent,name,size,pos,color,**extra):
        return ET.SubElement(parent,'geom',dict(name=name,type='box',size=' '.join(map(str,size)),
            pos=' '.join(map(str,pos)),rgba=color,friction='0.8 0.005 0.0001',**extra))
    geom(world,'task_table_top',[*half,.025],[*center,table_z-.025],'.58 .43 .26 1')
    for i,xy in enumerate(itertools.product((xmin+.02,xmax-.02),(ymin+.02,ymax-.02))):
        geom(world,f'table_leg_{i}',[.012,.012,(table_z-.05)/2],[*xy,(table_z-.05)/2],'.38 .4 .42 1')
    yaw = B['orientation_yaw_rad']; quat = [math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
    body = ET.SubElement(world,'body',dict(name='task_cube',pos=' '.join(map(str,cube)),quat=' '.join(map(str,quat))))
    ET.SubElement(body,'freejoint',name='task_cube_free')
    geom(body,'task_cube_geom',[.02]*3,[0,0,0],'.95 .45 .08 1',mass='.035',condim='4')
    tray = ET.SubElement(world,'body',dict(name='task_tray',pos=' '.join(map(str,[*tray_xy,table_z])),quat=' '.join(map(str,quat))))
    geom(tray,'tray_floor',[.07,.07,.004],[0,0,.004],'.1 .45 .8 1')
    for i,(size,pos) in enumerate((([.004,.074,.015],[-.074,0,.023]),([.004,.074,.015],[.074,0,.023]),
            ([.07,.004,.015],[0,-.074,.023]),([.07,.004,.015],[0,.074,.023]))):
        geom(tray,f'tray_wall_{i}',size,pos,'.12 .4 .7 1')
    for pose,color in [(A,'.2 .7 1 .65'),(B,'.2 .9 .3 .65'),(C,'.8 .3 .9 .65')]:
        ET.SubElement(world,'site',dict(name='new_pose_'+pose['id'],type='sphere',
            pos=' '.join(map(str,pose['position_world_m'])),size='.006',rgba=color,group='0'))
    return root

def center_layout(root, parameters):
    """Same final layout shift/leg correction/serialization as original code."""
    world = require(root,'worldbody')
    shift = np.asarray(parameters['layout_translation_world_m'],dtype=float)
    def get(tag,name): return require(world,f"{tag}[@name='{name}']")
    def vals(e,key): return np.fromstring(e.get(key),sep=' ')
    def put(e,key,values): e.set(key,' '.join(format(float(x),'.17g') for x in values))
    table = get('geom','task_table_top');put(table,'pos',vals(table,'pos')+shift)
    for name in ('task_cube','task_tray'):
        e=get('body',name);put(e,'pos',vals(e,'pos')+shift)
    underside=vals(table,'pos')[2]-vals(table,'size')[2]
    if underside <= 0:raise ValueError('Nonpositive table leg length')
    for i in range(4):
        e=get('geom','table_leg_'+str(i));p=vals(e,'pos');size=vals(e,'size')
        p[:2]+=shift[:2];p[2]=underside/2;size[2]=underside/2
        put(e,'pos',p);put(e,'size',size)
    root.set('model','g1_contact_centered_grasp_candidate')
    ET.indent(root,space='  ')
    # Original Windows Path.write_text used CRLF; make the frozen bytes portable.
    return ET.tostring(root,encoding='utf-8',xml_declaration=False).replace(b'\n', b'\r\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-xml',required=True,type=Path)
    parser.add_argument('--scene-xml',required=True,type=Path)
    parser.add_argument('--parameters',type=Path,default=Path(__file__).with_name('task_parameters.json'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    paths={'robot':args.robot_xml.resolve(),'scene':args.scene_xml.resolve()}
    raw={name:path.read_bytes() for name,path in paths.items()}
    for name,value in raw.items():
        if sha_bytes(value)!=INPUT_SHA[name]:raise ValueError('Pinned input hash mismatch: '+name)
    if args.output.exists() or (args.receipt and args.receipt.exists()):
        raise FileExistsError('Use fresh output/receipt paths; existing files are not overwritten')
    parameters=json.loads(args.parameters.read_text(encoding='utf-8-sig'))
    root=add_task_layout(source_tree(raw['robot'],raw['scene']),parameters)
    # Preserve the original intermediate serialize/parse boundary.
    output=center_layout(ET.fromstring(ET.tostring(root,encoding='unicode')),parameters)
    actual=sha_bytes(output)
    if actual!=parameters['expected_generated_scene_sha256']:
        raise ValueError('Generated scene differs from frozen scene: '+actual)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:stream.write(output)
    result=dict(status='generated_scene_byte_identical',output_sha256=actual,output_bytes=len(output),
        input_sha256=INPUT_SHA,upstream_revision=SOURCE_REVISION,python_version=sys.version.split()[0],numpy_version=np.__version__,
        generator_sha256=sha_bytes(Path(__file__).read_bytes()),parameters_sha256=sha_bytes(args.parameters.read_bytes()),
        physics_steps=0,viewer_opened=False,downloads=0,original_inputs_modified=False,
        licensing_note='Local generation does not establish permission to redistribute the supplied or generated XML.')
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        with args.receipt.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
