<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalMaxPooling3D extends AbstractGlobalMaxPooling
{
    protected $rank = 3;

    public function __construct(object $backend,array $options=null)
    {
        $leftargs = [];
        extract($this->extractArgs([
            'name'=>null,
        ],$options,$leftargs));
        $this->initName($name,'globalmaxpooling3d');
        parent::__construct($backend, $leftargs);
    }
}
